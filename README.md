# Ecoscout 后端运作说明与前端接口文档

## 1. 文档目的

这份 README 不再以项目展示为主，而是面向团队开发协作，重点回答两个问题：

1. 后端是如何启动、组织和处理检测任务的。
2. 前端应该如何对接接口、理解字段、轮询任务和展示结果。

如果你是后端同学，可以把它当成系统设计和运行说明。  
如果你是前端同学，可以直接从“前端对接接口说明”一节开始接入。

---

## 2. 后端总体定位

Ecoscout 是一个以 FastAPI 为核心的视觉检测后端，负责完成以下工作：

- 接收图片、Base64 图像和视频上传。
- 调用目标检测模型识别垃圾桶、垃圾溢出、散落垃圾、火焰、烟雾。
- 对垃圾桶进一步做颜色分类，补充垃圾桶类型语义。
- 将图片检测产生的预警记录、检测明细、视频任务状态写入 SQLite。
- 对视频任务进行异步处理，并持续向前端暴露进度、当前报警状态和最终结果视频。
- 通过 Jinja2 模板直接提供内置页面，同时也能作为纯 API 后端供独立前端调用。

当前后端没有认证、权限和 CORS 中间件，默认假设前后端同源部署。如果后续前端要分离部署，需要额外补充跨域配置。

---

## 3. 技术栈与运行角色

### 3.1 核心框架

- `FastAPI`：HTTP API 和页面路由入口。
- `Uvicorn`：ASGI 服务启动器。
- `Pydantic v2`：请求和响应结构校验。
- `SQLAlchemy`：SQLite ORM。
- `Jinja2`：服务端模板页面。

### 3.2 视觉推理

- `onnxruntime`：优先使用的推理后端。
- `Ultralytics YOLO`：当 ONNX 模型不可用时回退到 `.pt` 权重。
- `OpenCV`：图片解码、视频逐帧读取、绘框、视频处理。
- `PyTorch` / `TorchVision`：垃圾桶颜色分类模型加载。

### 3.3 异步能力

- `Celery + Redis`：标准异步视频任务链路。
- 本地线程兜底：当 Redis/Celery worker 不可用时，后端仍可直接在本机线程中处理视频任务，保证演示和联调可继续进行。

---

## 4. 代码结构与职责划分

```text
app/
├─ api/
│  ├─ pages.py              # 页面路由
│  └─ routes.py             # API 路由
├─ services/
│  ├─ detection_service.py  # 图片/摄像头检测主流程
│  ├─ video_service.py      # 视频逐帧处理、事件状态管理
│  ├─ record_service.py     # 记录入库、列表、详情、统计
│  ├─ inference.py          # ONNX / Ultralytics 双后端封装
│  └─ bin_color_service.py  # 垃圾桶颜色分类
├─ upgrade/
│  ├─ pipeline.py           # 检测、跟踪、报警组合流水线
│  ├─ tracker.py            # 跟踪逻辑
│  ├─ alarm.py              # 报警判定
│  └─ detection.py          # 检测适配
├─ templates/               # 内置页面模板
├─ models/                  # 模型文件
├─ main.py                  # FastAPI 应用入口
├─ config.py                # 配置与环境变量
├─ bootstrap.py             # 启动初始化
├─ database.py              # 数据库引擎和 Session
├─ db_models.py             # ORM 模型
├─ schemas.py               # Pydantic 响应模型
├─ dependencies.py          # 依赖注入
├─ celery_app.py            # Celery 应用
└─ tasks.py                 # 视频任务执行入口
```

---

## 5. 后端启动原理

### 5.1 应用启动入口

启动入口是 `app/main.py`。

`create_app()` 做了几件关键事情：

1. 调用 `bootstrap_application()`。
2. 创建 `FastAPI` 实例。
3. 注册统一异常处理：
   - `HTTPException` 返回 `{ "error": ... }`
   - 请求参数校验失败返回 `{ "error": "请求参数校验失败", "detail": [...] }`
4. 初始化 `Jinja2Templates`。
5. 将 `app/uploads` 挂载到静态路径 `/uploads`。
6. 注册页面路由和 API 路由。

### 5.2 启动初始化做了什么

`bootstrap_application()` 在 `app/bootstrap.py` 中完成：

- 创建上传目录 `app/uploads/alerts` 和 `app/uploads/videos`
- 自动建表
- 检查 `video_task_records` 是否存在 `runtime_state` 字段，不存在就补列

这意味着项目第一次启动时不需要手动建库，数据库会自动初始化。

### 5.3 配置来源

配置在 `app/config.py` 中定义，默认读取根目录 `.env`。

几个关键事实：

- 默认数据库：`sqlite:///garbage_system.db`
- 默认 API 前缀：`/api`
- 默认 Redis：`redis://localhost:6379/0`
- 默认视频抽帧参数：`VIDEO_DEFAULT_SKIP_FRAMES=1`
- 默认上传目录：`app/uploads`

`.env.example` 目前主要提供了模型路径和阈值示例，但 `config.py` 中定义的所有字段都可以通过 `.env` 覆盖。

---

## 6. 系统运行链路

### 6.1 图片上传检测链路

接口：`POST /api/detect/image`

完整流程如下：

1. 前端以 `multipart/form-data` 上传图片文件，字段名必须是 `file`。
2. 路由层检查扩展名，只允许图片格式。
3. 使用 OpenCV 解码为 `numpy.ndarray`。
4. 通过依赖注入获取全局缓存的 `DetectionService`。
5. `DetectionService.detect()` 执行检测：
   - 优先用 ONNX 模型
   - ONNX 不可用时回退到 Ultralytics `.pt`
   - 如果检测模型都不可用，进入演示模式，返回伪造检测结果，保证前端可联调
6. 如果识别到垃圾桶，并且颜色分类模型已加载，会继续做垃圾桶颜色分类，补出：
   - `bin_color`
   - `bin_type_key`
   - `bin_type_name`
7. 对溢出和散落垃圾结果，后端会关联最近的垃圾桶类型，补出：
   - `related_bin_type_key`
   - `related_bin_type_name`
8. `UpgradePipeline` 给检测结果附加 `track_id`，便于连续请求保持追踪语义。
9. 后端绘制结果图，并编码成 Base64 返回给前端。
10. 如果场景中存在预警，调用 `RecordService.create_alert_record()`：
    - 保存标注后图片
    - 创建 `alert_records`
    - 为每个检测框创建 `detection_records`

注意：

- 图片上传和 Base64 检测默认“不做冷却抑制”，每次请求都会即时判断是否报警。
- 只有 `scene.alert_count > 0` 才会落库生成预警记录。

### 6.2 摄像头 Base64 检测链路

接口：`POST /api/detect/base64`

这条链路与图片上传非常接近，区别在于：

- 请求体是 JSON：`{ "image": "data:image/jpeg;base64,..." }`
- 后端先把 Base64 解码成帧，再进入同样的检测流程
- 记录来源 `source` 会写成 `camera`

前端页面当前做法是每秒抓拍一次摄像头帧并调用该接口。

### 6.3 视频检测链路

接口：`POST /api/detect/video`

视频任务不是同步返回最终结果，而是“提交任务 + 轮询状态”模式。

提交流程：

1. 前端上传视频文件，字段名 `file`。
2. 可附带 `skip_frames`，控制抽帧处理间隔。
3. 后端生成 `task_id`。
4. 视频原文件先落到 `app/uploads/videos/`。
5. 在 `video_task_records` 表中写入一条 `pending` 任务记录。
6. 后端判断 Celery worker 是否可用：
   - 可用：通过 `process_video_task.apply_async()` 投递到 Celery
   - 不可用：启动本地线程执行 `run_video_task()`

执行流程在 `app/tasks.py` 和 `app/services/video_service.py` 中：

1. 打开视频，读取总帧数、帧率、宽高。
2. 按 `skip_frames` 抽帧检测。
3. 每个被处理的帧执行：
   - 目标检测
   - 跟踪与 `track_id` 附加
   - 事件确认与持续状态管理
   - 抑制低优先级重叠告警
   - 在画面上绘框和绘制“当前激活报警”面板
4. 持续回写任务表：
   - `status`
   - `progress`
   - `message`
   - `runtime_state`
5. 处理完成后输出 `*_detected.mp4` 到 `app/uploads/videos/`
6. 生成一条视频预警汇总记录到 `alert_records`
7. 更新任务状态为 `completed`
8. 删除原始上传视频

### 6.4 视频事件状态管理的核心含义

视频检测不是简单的“每帧有框就报一次”，后端专门实现了一套事件状态管理：

- 火焰、烟雾：`2-frame / 1-hit` 的快速确认
- 溢出、散落垃圾：`3-frame continuous` 的连续确认
- 支持事件持续、结束、冷却、优先级抑制
- `active_alerts` 反映当前仍在持续中的事件
- `new_alert_count` 表示本轮新触发
- `sustained_alert_count` 表示仍在持续提醒
- `ended_alert_count` 表示刚结束

这也是前端视频页需要轮询展示“当前激活报警”的原因。

---

## 7. 检测与模型策略

### 7.1 统一类别

后端对外统一输出 5 类主类别：

| class_id | 英文键 | 说明 | 是否预警 |
| --- | --- | --- | --- |
| 0 | `garbage_bin` | 垃圾桶 | 否 |
| 1 | `overflow` | 垃圾溢出 | 是 |
| 2 | `garbage` | 散落垃圾 | 是 |
| 3 | `fire` | 火焰 | 是 |
| 4 | `smoke` | 烟雾 | 是 |

### 7.2 模型后端选择策略

后端在 `app/services/inference.py` 中统一封装了两类推理后端：

- `OnnxYoloBackend`
- `UltralyticsBackend`

策略是：

1. 优先加载 ONNX
2. ONNX 加载失败时，回退到 `.pt`
3. 如果两者都不可用，则该类别模型视为未加载

### 7.3 检测后处理

`DetectionService` 里还做了多层后处理，不只是原始模型输出：

- 置信度筛选
- 最小框面积筛选
- 火焰误报抑制
- 烟雾误报抑制
- 火焰对垃圾类告警的优先级覆盖
- 按类别做额外 NMS
- 垃圾桶颜色补分类
- 溢出/散落垃圾与最近垃圾桶关联

### 7.4 场景状态判定

后端返回的 `scene.status` 不是任意字符串，而是以下集合之一：

- `normal`
- `warning`
- `overflow`
- `smoke`
- `fire`

优先级是：

1. 只要出现火焰，场景状态优先记为 `fire`
2. 否则如果出现烟雾，记为 `smoke`
3. 否则如果出现溢出，记为 `overflow`
4. 否则只要有任何预警类，记为 `warning`
5. 都没有则为 `normal`

---

## 8. 数据库存储与文件存储

### 8.1 数据表

#### `alert_records`

用于存储每次预警的主记录：

- `record_uid`：前端展示和详情查询用的记录 ID
- `created_at`：记录创建时间
- `status`：场景状态
- `alert_types`：预警类型数组
- `total_detections`：本次检测总框数
- `alert_count`：本次预警框数
- `result_image_path`：图片记录对应截图路径，视频记录则可能是 `video_task:{task_id}`
- `source`：`image` / `camera` / `video`

#### `detection_records`

用于存储单个预警记录下每个检测框的明细：

- `class_id`
- `class_name`
- `confidence`
- `bbox`
- `is_alert`
- `source_model`

#### `video_task_records`

用于存储视频任务生命周期：

- `task_id`
- `status`
- `input_filename`
- `input_path`
- `output_path`
- `progress`
- `message`
- `total_frames`
- `detected_frames`
- `total_detections`
- `total_alerts`
- `video_info`
- `runtime_state`
- `error_detail`

注意：

- 当前实现里的 `detected_frames` 更接近“存在激活告警的帧数”，不是“所有被处理帧数”。
- `runtime_state` 是一个 JSON 字符串，用于给前端实时展示激活告警状态。

### 8.2 文件目录

| 路径 | 用途 |
| --- | --- |
| `app/uploads/alerts/` | 图片/摄像头预警截图 |
| `app/uploads/videos/` | 上传视频与处理后视频 |
| `/uploads/...` | 前端访问静态结果文件的 URL 前缀 |

前端拿到 `result_video` 后，需要自己拼成 `/uploads/${result_video}` 才能播放。

---

## 9. 页面路由说明

这些页面由后端直接渲染：

| 路由 | 说明 |
| --- | --- |
| `/` | 首页 |
| `/detection` | 综合检测页，支持图片、摄像头、视频 |
| `/video` | 独立视频检测页 |
| `/alerts` | 预警记录页 |
| `/statistics` | 统计页 |
| `/dataset` | 数据集说明页 |
| `/docs` | FastAPI 自动文档 |

额外说明：

- 代码里还注册了 `/collection` 页面路由，但当前仓库没有对应 `collection.html` 模板，可视为占位或未完成页面。

---

## 10. 前端对接接口说明

### 10.1 通用约定

#### 成功响应

不同接口字段不同，但多数都会包含 `success: true` 或明确的数据对象。

#### 错误响应

后端统一错误格式主要有两种：

```json
{ "error": "Task not found" }
```

```json
{
  "error": "请求参数校验失败",
  "detail": [
    {
      "loc": ["body", "image"],
      "msg": "Field required",
      "type": "missing"
    }
  ]
}
```

前端处理建议：

- 优先读 `error`
- 没有 `error` 时再读 `detail`
- 对视频接口再兜底读取 `message`

#### Base64 约定

- 请求 `/api/detect/base64` 时：传完整 Data URL，形如 `data:image/jpeg;base64,...`
- 响应里的 `result_image` 和 `/api/alerts/{id}/image` 返回的 `image`：是不带前缀的纯 Base64 字符串

前端展示时应拼接：

```js
`data:image/jpeg;base64,${base64String}`
```

#### 视频结果路径约定

`result_video` 返回的是相对路径，不带 `/uploads/` 前缀。

前端播放时应拼接：

```js
`/uploads/${result_video}`
```

---

### 10.2 图片检测接口

#### `POST /api/detect/image`

##### 请求

- `Content-Type: multipart/form-data`
- 字段：
  - `file`: 图片文件，必填

##### 响应

```json
{
  "success": true,
  "detections": [
    {
      "class_id": 0,
      "class_name": "可回收垃圾桶",
      "confidence": 0.92,
      "bbox": [120, 55, 320, 460],
      "alert": false,
      "icon": "",
      "source": "garbage",
      "track_id": 3,
      "bin_color": "blue",
      "bin_color_confidence": 0.95,
      "bin_type_key": "recyclable",
      "bin_type_name": "可回收垃圾桶",
      "related_bin_type_key": null,
      "related_bin_type_name": null
    }
  ],
  "scene": {
    "status": "normal",
    "alert_count": 0,
    "alert_types": [],
    "normal_count": 1,
    "total": 1,
    "timestamp": "2026-06-05 14:20:10"
  },
  "result_image": "<base64>"
}
```

##### 字段说明

| 字段 | 说明 |
| --- | --- |
| `detections` | 每个检测框的结果数组 |
| `bbox` | `[x1, y1, x2, y2]`，像素坐标 |
| `alert` | 当前框是否属于预警项 |
| `source` | 产生该框的模型来源，例如 `garbage` / `fire` / `smoke` / `demo` |
| `track_id` | 跟踪 ID，图片和 Base64 连续请求时可用于前端做去抖或追踪展示 |
| `bin_color` | 垃圾桶颜色分类结果，仅 `class_id=0` 时可能存在 |
| `bin_type_key` / `bin_type_name` | 垃圾桶细分类 |
| `related_bin_type_key` / `related_bin_type_name` | 对溢出和散落垃圾补充的最近垃圾桶类型 |
| `scene` | 整张图层面的汇总信息 |
| `result_image` | 绘框后的图片，纯 Base64 |

##### 前端建议

- 上传后直接使用 `result_image` 回显，不需要前端自己再绘框。
- 如果要做结构化卡片展示，优先读 `class_name`，而不是自己用 `class_id` 映射中文。
- 如果是垃圾桶卡片，优先显示 `bin_type_name`。
- 如果是溢出/散落垃圾，优先显示 `related_bin_type_name` 以帮助用户知道关联的是哪类垃圾桶。

---

### 10.3 摄像头 Base64 检测接口

#### `POST /api/detect/base64`

##### 请求

```json
{
  "image": "data:image/jpeg;base64,..."
}
```

##### 响应

响应结构与 `POST /api/detect/image` 完全一致。

##### 前端建议

- 当前内置页面每秒发送一次抓拍请求，这是合理轮询频率。
- 由于后端会返回 `track_id`，前端可以选择：
  - 直接覆盖上一帧结果
  - 或按 `track_id` 做轻量追踪动画

---

### 10.4 视频任务提交接口

#### `POST /api/detect/video`

##### 请求

- `Content-Type: multipart/form-data`
- 字段：
  - `file`: 视频文件，必填
  - `skip_frames`: 抽帧间隔，可选，最小实际生效值为 1

##### 响应

```json
{
  "success": true,
  "task_id": "d8f819f0b8fc4b6ba6f7f6f58f412345",
  "status": "pending",
  "message": "Task queued for background processing"
}
```

##### 前端建议

- 一旦拿到 `task_id`，立即开始轮询 `/api/tasks/{task_id}`。
- 当前模板页使用约 `1200ms` 一次轮询，可直接沿用。
- 如果用户离开页面，建议把 `task_id`、最近进度、最近 `active_alerts` 暂存到 `localStorage`，回来后恢复。

---

### 10.5 视频任务状态接口

#### `GET /api/tasks/{task_id}`

##### 响应字段

```json
{
  "success": true,
  "task_id": "d8f819f0b8fc4b6ba6f7f6f58f412345",
  "status": "processing",
  "progress": 42,
  "message": "视频处理中 42%",
  "result_video": null,
  "stats": null,
  "active_alerts": [
    {
      "class_id": 3,
      "class_name": "火焰",
      "priority": 0,
      "state": "new",
      "duration_seconds": 1.6,
      "confirmation_mode": "2-frame / 1-hit"
    }
  ],
  "active_alert_count": 1,
  "highest_priority_alert": "火焰",
  "new_alert_count": 1,
  "sustained_alert_count": 0,
  "ended_alert_count": 0
}
```

当 `status === "completed"` 时，`stats` 和 `result_video` 会返回：

```json
{
  "success": true,
  "task_id": "d8f819f0b8fc4b6ba6f7f6f58f412345",
  "status": "completed",
  "progress": 100,
  "message": "视频处理完成",
  "result_video": "videos/d8f819f0b8fc4b6ba6f7f6f58f412345_detected.mp4",
  "stats": {
    "total_frames": 1280,
    "detected_frames": 143,
    "total_detections": 512,
    "total_alerts": 16,
    "suppressed_alerts": 9,
    "alert_types": ["火焰", "烟雾"],
    "video_info": "1920x1080, 30.0fps, suppressed=9",
    "active_alerts": [],
    "active_alert_count": 0,
    "highest_priority_alert": null,
    "new_alert_count": 0,
    "sustained_alert_count": 0,
    "ended_alert_count": 0
  },
  "active_alerts": [],
  "active_alert_count": 0,
  "highest_priority_alert": null,
  "new_alert_count": 0,
  "sustained_alert_count": 0,
  "ended_alert_count": 0
}
```

##### `status` 枚举

- `pending`
- `processing`
- `completed`
- `failed`

##### 前端建议

- 进度条读 `progress`
- 文案读 `message`
- “当前激活报警”面板读 `active_alerts`
- “最高优先级报警”读 `highest_priority_alert`
- 视频播放地址读 `result_video`，前端拼成 `/uploads/${result_video}`
- 如果 `status=failed`，直接展示 `message`

---

### 10.6 预警记录列表接口

#### `GET /api/alerts`

##### 查询参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `page` | int | `1` | 页码 |
| `per_page` | int | `20` | 每页数量 |
| `status` | string | `all` | `all` / `warning` / `normal` |

##### 重要语义

- `status=warning` 在后端里表示“筛选所有 `status != normal` 的记录”
- 它不是只筛选字面值为 `warning` 的记录

##### 响应

```json
{
  "total": 58,
  "page": 1,
  "per_page": 10,
  "records": [
    {
      "id": "7ac31f2b",
      "time": "2026-06-05 14:32:10",
      "status": "fire",
      "types": ["火焰"],
      "total": 3,
      "alert_count": 1,
      "source": "image"
    }
  ]
}
```

##### 前端建议

- 列表页直接用 `records`
- 缩略图如果 `source !== "video"`，再调用 `/api/alerts/{id}/image`
- 如果 `source === "video"`，详情页直接走 `/api/alerts/{id}/detail`

---

### 10.7 预警截图接口

#### `GET /api/alerts/{record_uid}/image`

##### 响应

```json
{
  "image": "<base64>"
}
```

##### 前端建议

- 只适用于图片/摄像头类记录
- 视频类记录通常没有单张截图，而是视频详情

---

### 10.8 预警详情接口

#### `GET /api/alerts/{record_uid}/detail`

这个接口是前端查看详情时最关键的统一入口。

##### 图片类记录返回示例

```json
{
  "id": "7ac31f2b",
  "source": "image",
  "status": "fire",
  "types": ["火焰"],
  "alert_count": 1,
  "total_detections": 3,
  "time": "2026-06-05 14:32:10",
  "detail_type": "image",
  "image": "<base64>"
}
```

##### 视频类记录返回示例

```json
{
  "id": "vd8f819f",
  "source": "video",
  "status": "warning",
  "types": ["火焰", "烟雾"],
  "alert_count": 16,
  "total_detections": 512,
  "time": "2026-06-05 14:50:10",
  "detail_type": "video",
  "task_id": "d8f819f0b8fc4b6ba6f7f6f58f412345",
  "result_video": "videos/d8f819f0b8fc4b6ba6f7f6f58f412345_detected.mp4",
  "stats": {
    "total_frames": 1280,
    "detected_frames": 143,
    "total_detections": 512,
    "total_alerts": 16,
    "suppressed_alerts": 9,
    "video_info": "1920x1080, 30.0fps, suppressed=9",
    "active_alerts": [],
    "active_alert_count": 0,
    "highest_priority_alert": null,
    "new_alert_count": 0,
    "sustained_alert_count": 0,
    "ended_alert_count": 0
  }
}
```

##### 前端建议

- 优先判断 `detail_type`
- `detail_type=image`：展示图片
- `detail_type=video`：展示视频统计信息 + “打开检测视频”按钮

---

### 10.9 统计接口

#### `GET /api/statistics`

##### 响应

```json
{
  "total_detections": 1234,
  "total_alerts": 218,
  "today_alerts": 17,
  "hourly_alerts": [0, 0, 1, 0, 0, 0, 2, 1, 0, 0, 3, 5, 2, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
  "class_stats": [
    {
      "class_id": 3,
      "class_name": "火焰",
      "count": 80,
      "is_alert": true
    }
  ],
  "start_time": "2026-06-05 13:10:02",
  "alert_record_count": 72
}
```

##### 字段说明

| 字段 | 说明 |
| --- | --- |
| `total_detections` | 所有检测明细总数 |
| `total_alerts` | 所有预警记录中的 `alert_count` 累加值 |
| `today_alerts` | 今天新增的预警记录数，不是今天的检测框数 |
| `hourly_alerts` | 全天 24 小时预警记录分布 |
| `class_stats` | 按类别汇总的检测明细统计 |
| `start_time` | 当前 FastAPI 进程启动时间 |
| `alert_record_count` | 预警主记录数 |

##### 前端建议

- 仪表盘大盘数字：读 `total_detections` / `total_alerts`
- 柱状图：读 `class_stats`
- 24 小时分布：读 `hourly_alerts`
- 系统运行时长：前端自己用 `start_time` 与当前时间相减

---

### 10.10 类别元数据接口

#### `GET /api/classes`

##### 响应

```json
{
  "classes": [
    {
      "id": 0,
      "name": "垃圾桶",
      "en": "garbage_bin",
      "alert": false,
      "icon": ""
    }
  ],
  "bin_types": {
    "recyclable": {
      "name": "可回收垃圾桶",
      "color": "#2196F3",
      "classes": [0]
    }
  }
}
```

##### 前端建议

- 如果要把类别映射逻辑做成可配置而不是写死在前端，可以在应用启动后先拉一次这个接口。

---

### 10.11 系统状态接口

#### `GET /api/status`

##### 响应

```json
{
  "model_loaded": true,
  "garbage_model": true,
  "fire_model": true,
  "smoke_model": true,
  "bin_color_model": true,
  "mode": "正常检测",
  "uptime": "2026-06-05 13:10:02",
  "class_count": 5,
  "version": "2.0.0",
  "name": "社区垃圾与火情识别预警系统"
}
```

##### 说明

- `model_loaded=false` 时，通常意味着主检测模型不可用，后端会落入演示模式。
- 前端可用它来做健康检查页、调试面板或启动提示。

---

## 11. 依赖注入与对象生命周期

这部分对后端开发和前端理解响应一致性都很重要。

### 11.1 `DetectionService` 生命周期

`get_detection_service()` 使用了 `@lru_cache`，因此在 FastAPI Web 进程里：

- `DetectionService` 通常只初始化一次
- 模型加载只发生一次
- 后续 HTTP 请求复用同一个检测服务实例

这带来的影响是：

- 图片和摄像头接口共享同一套模型实例
- 共享同一个 `UpgradePipeline` 追踪上下文

### 11.2 视频任务中的服务实例

视频任务执行时不会直接复用 Web 进程里的依赖实例，而是在任务执行函数中重新创建：

- `DetectionService`
- `VideoProcessingService`
- `RecordService`

这样可以让 Celery worker 独立运行，不依赖 Web 请求上下文。

---

## 12. 本地运行方式

### 12.1 Windows 一键启动

项目自带 `start_queue.bat`。

它会执行以下动作：

1. 查找项目里的虚拟环境目录，优先 `.venv`、`.venv311`、`venv`
2. 检查关键依赖是否已安装
3. 清理旧的 Uvicorn / Celery Python 进程
4. 在后台启动 Celery worker
5. 在后台启动 FastAPI
6. 自动打开浏览器

默认访问地址是：

```text
http://127.0.0.1:8010
```

注意：

- 这个脚本假设依赖已经预装在虚拟环境里
- 它不会自动创建虚拟环境，也不会现场 `pip install`

### 12.2 手动启动 Web

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 12.3 手动启动 Celery worker

```bash
python -m celery -A app.celery_app worker --loglevel=info --pool=solo
```

如果不启动 Celery worker，视频任务仍可运行，但会回退到本地线程模式。

---

## 13. 推荐环境变量

下面是当前项目最值得团队关注的配置：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `APP_NAME` | `社区垃圾与火情识别预警系统` | 应用名 |
| `APP_VERSION` | `2.0.0` | 版本号 |
| `DEBUG` | `false` | 调试模式 |
| `DATABASE_URL` | `sqlite:///garbage_system.db` | SQLite 数据库地址 |
| `REDIS_URL` | `redis://localhost:6379/0` | Celery broker/backend |
| `CELERY_TASK_ALWAYS_EAGER` | `false` | Celery 同步执行开关 |
| `VIDEO_DEFAULT_SKIP_FRAMES` | `1` | 视频默认抽帧间隔 |
| `GARBAGE_ONNX_MODEL` | `app/models/garbege.onnx` | 垃圾检测 ONNX |
| `FIRE_ONNX_MODEL` | `app/models/only_fire.onnx` | 火焰检测 ONNX |
| `SMOKE_ONNX_MODEL` | `.env.example` 中给出示例 | 烟雾检测 ONNX |
| `BIN_COLOR_RESNET18_MODEL` | `app/models/bin_color_resnet18.pt` | 垃圾桶颜色分类模型 |

---

## 14. 给前端团队的接入建议

### 14.1 最简单接法

如果你们只是接现有后端：

1. 图片页接 `POST /api/detect/image`
2. 摄像头页接 `POST /api/detect/base64`
3. 视频页接 `POST /api/detect/video` + `GET /api/tasks/{task_id}`
4. 历史记录页接 `GET /api/alerts`
5. 详情弹窗接 `GET /api/alerts/{id}/detail`
6. 统计页接 `GET /api/statistics`

### 14.2 展示字段优先级

- 检测名称优先显示 `class_name`
- 图片优先显示 `result_image`
- 视频优先显示 `/uploads/${result_video}`
- 当前告警优先读 `active_alerts`
- 告警原因优先读 `scene.alert_types` 或详情里的 `types`

### 14.3 前端需要特别注意的坑

- `/api/detect/base64` 请求里要传带前缀的 Data URL
- 返回的 `result_image` / `image` 不带 Data URL 前缀
- `status=warning` 不是字面值过滤，而是“所有非 normal”
- `today_alerts` 表示今天的预警记录数，不是今天的检测框数
- `detected_frames` 当前实现表示有激活告警的帧数，语义不要误解
- 当前没有取消视频后端任务的接口，前端“停止检测”只能停止轮询，后台任务可能继续执行

### 14.4 如果以后要做前后端分离

建议后续补这些基础设施：

- CORS 中间件
- 认证与用户体系
- 统一错误码
- OpenAPI 导出后的前端类型生成
- 视频任务取消接口
- WebSocket 或 SSE 实时推送，替代轮询

---

## 15. 结论

这个项目的后端已经不是单纯的“上传图片返回识别结果”，而是一个完整的检测服务：

- 有启动初始化
- 有模型加载与回退策略
- 有图片/摄像头实时检测
- 有视频异步处理与任务状态轮询
- 有记录、详情、统计和静态结果文件访问

对于前端来说，最核心的契约是三条：

1. 图片和摄像头接口直接返回可展示结果图和结构化检测框。
2. 视频接口采用“提交任务 + 轮询任务状态 + 播放结果视频”的模式。
3. 历史记录、详情和统计接口已经足够支撑完整管理端页面。

如果后续团队继续扩展，建议优先在“统一错误码、CORS、任务取消、实时推送”这四个方向补强。
