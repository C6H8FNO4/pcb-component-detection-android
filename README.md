# PCB 焊接/装配缺陷检测 Android App（v2.3）

完全离线的 Android 应用：模型打包在 APK 里，用 ONNX Runtime 在手机上直接推理，**不需要网络、不需要服务器**，照片不出设备。

| 项目 | 值 |
|---|---|
| 模型 | **YOLO26m**（模型卡：`ParsaVictor/pcb-solder-defect-detection-yolo26`） |
| 参数 / 算力 | 20,356,391 / 68.1 GFLOPs |
| 推理尺寸 | **800**（与作者训练设置一致） |
| 导出物 | ONNX fp32 **77.9 MiB (81.7 MB)**，opset 13，输出 `(1, 13, 13125)` |
| 任务 | 检测（9 类），**需自行解码 + NMS** |
| 应用标识 | 包名 `com.c6h8fno4.PcbSolderDefectDetection`，显示名「PCB焊接缺陷检测」 |

### 关于输入尺寸（640 vs 800）

作者训练用的是 **imgsz 800**（150 epochs）。我曾在 687 张 val 上试过 640，精度不降反微升：

| 推理尺寸 | mAP50 | mAP50-95 | P | R | 桌面 CPU 耗时 |
|---|---|---|---|---|---|
| **800（当前采用，作者设置）** | 0.9107 | 0.6350 | 0.9105 | 0.9147 | 450 ms |
| 640（备选） | 0.9151 | 0.6409 | 0.8973 | 0.9048 | 285 ms |

**v2.2 已改回 800**，与作者配置保持一致；代价是单帧推理从 ~578 ms 回到 ~1050 ms。
如果哪天更看重速度、可以接受这个差异，把 `Detector.INPUT_SIZE` 改成 640、并把 `assets/best.onnx` 换成 `webapp\models\solder_yolo26m_640.onnx` 即可。

> `150 epochs` 是训练参数，权重定型后对推理没有影响 —— App 里只有 `imgsz` 起作用。

## 1. 9 个类别与阈值

置信度用模型卡给定的**逐类 F1 最优阈值**（比统一阈值准），NMS IoU 固定 0.45。

| # | 类别 | 中文 | 阈值 |
|---|---|---|---|
| 0 | insufficient_solder | 少锡 | 0.297 |
| 1 | excess_solder | 多锡 | 0.349 |
| 2 | solder_bridge | 连锡 | 0.417 |
| 3 | solder_void | 空洞 | 0.249 |
| 4 | missing_component | 缺件 | 0.381 |
| 5 | shifted_component | 偏移 | 0.389 |
| 6 | burn | 烧焦 | 0.224 |
| 7 | physical_damage | 损伤 | 0.214 |
| 8 | oxidation | 氧化 | 0.228 |

改阈值：`Detector.CONF_PER_CLASS`；想用一个统一阈值就调 `detector.detect(bmp, conf, iou)` 那个重载。

---

## 2. 安装

| APK | 体积 | 用在哪 |
|---|---|---|
| `pcb-defect-solder-v2.3-arm64.apk` | **108 MB** | **真机（99% 的手机）** |
| `app-x86_64-debug.apk` | 114 MB | 模拟器 |
| `app-universal-debug.apk` | 140 MB | 不确定架构时用 |

```bash
adb install -r pcb-defect-solder-v2.0-arm64.apk
```

包名 `com.c6h8fno4.PcbSolderDefectDetection`，versionCode 5 / versionName 2.3，minSdk 24，targetSdk 36，debug 证书签名（自用/内测；对外发布需换自有 keystore）。

> APK 为什么大：模型本体 78 MB + ONNX Runtime 原生库约 20 MB + 界面/CameraX。
> 想压到 ~30 MB 需要做 int8 量化（见第 6 节），但要重新验证精度。

---

## 3. 两个模式

### 单张检测（相册 / 拍照）—— 主力用法

- **从相册选图**：系统文件选择器（SAF），无需存储权限
- **拍照**：系统相机，原图存应用缓存目录（FileProvider 授权）

结果给出：标注图、按类别计数、预处理/推理/后处理耗时、每个框的类别+置信度+像素坐标。

### 实时检测 —— 预览流畅，检测框低频刷新

**v2.1 起显示与推理解耦**：相机每来一帧就立刻转成 Bitmap、画上「上一次的检测结果」直接显示；
推理丢到独立线程异步跑，算完再换框。所以预览帧率不再被推理（700 ms+）拖住，
只有**框**的更新频率受限于推理速度。

因为画框和显示用的是同一张 Bitmap，坐标天然对齐，不需要任何预览裁剪映射。

**实测（模拟器 x86_64，软件渲染，4 核）**：预览 **9–19 fps**（v2.0 是 1.3 fps），推理 578–743 ms/帧。
真机（arm64，8 核）预览应该接近满帧，推理估 300–500 ms，即框大约每秒刷新 2–3 次。

HUD 上会同时显示「预览 fps」和「推理 ms」，可以直观区分是**画面卡**还是**框更新慢**。

想再快只能降模型成本：int8 量化（预计再快 2 倍左右）或换更小的骨干网络。

---

## 4. 实测数据（模拟器 x86_64，软件渲染，无 GPU）

| 项目 | v2.0（800 / 同步显示） | **v2.1（640 / 异步显示）** |
|---|---|---|
| 模型加载 | 1.98 s | 约 4.2 s（78 MB 模型） |
| 单张预处理 | 362 ms | **481 ms** |
| 单张推理 | 1329 ms（冷）/ 1036 ms | **1050 ms（800）** |
| 单张后处理 | 21 ms | **1 ms** |
| 实时预览帧率 | **1.3 fps** | **9–19 fps** |
| 实时推理 | 707–1036 ms | 578–743 ms |
| 精度一致性 | App 0.814/0.792 vs Python 0.815/0.793 ✅ | App 0.752/0.744，同样 2 个 `excess_solder` ✅ |

桌面 ORT 基准（6 线程，供对比）：

| 输入 | ONNX | 输出 | 中位耗时 |
|---|---|---|---|
| 800 | 81.8 MB | (1, 13, 13125) | 450 ms |
| **640（采用）** | 81.7 MB | (1, 13, 8400) | **285 ms** |

---

## 5. 源码结构

```
android/
├─ settings.gradle / build.gradle / gradle.properties / local.properties
└─ app/
   ├─ build.gradle                  # compileSdk 36 / minSdk 24 / ABI 拆包
   └─ src/main/
      ├─ AndroidManifest.xml
      ├─ assets/best.onnx           # 77.99 MB，YOLO26m 800
      ├─ java/com/pcb/defect/
      │  ├─ Detector.java           # ★ 核心：letterbox(800) + ORT + 解码 + 逐类阈值 NMS
      │  ├─ BoxRenderer.java        # 画框，9 类各一色
      │  ├─ MainActivity.java       # 单张检测
      │  └─ LiveActivity.java       # 实时检测（CameraX）
      └─ res/
```

### 核心实现要点（`Detector.java`）

1. **预处理**：等比缩放到 800 长边，灰色 114 居中填充，RGB、`/255`、CHW
2. **推理**：ORT，线程数 = CPU 核数 / 2
3. **解码**：`cx,cy,w,h` 是 800 输入坐标系的像素 → 减 padding、除 scale 还原到原图
4. **NMS**：逐类别贪心，每类先按**自己的阈值**过滤，候选 >300 个先截断

### 换模型的步骤

1. `yolo export model=best.pt format=onnx imgsz=800 opset=13 simplify=False`
2. 覆盖 `app/src/main/assets/best.onnx`
3. 改动：`Detector.INPUT_SIZE` / `NUM_CLASSES` / `LABELS` / `LABELS_CN` / `CONF_PER_CLASS`，以及 `BoxRenderer.COLORS`
4. 重新编译

---

## 6. 重新编译

```powershell
$env:JAVA_HOME        = "C:\Program Files\Android\Android Studio\jbr"
$env:ANDROID_HOME     = "$env:LOCALAPPDATA\Android\Sdk"
$env:GRADLE_USER_HOME = "<repo>\android\.gradle-home"
cd android
.\gradle-9.6.0\bin\gradle.bat :app:assembleDebug
```

产物（ABI 已拆包）：`app/build/outputs/apk/debug/`
`app-arm64-v8a-debug.apk`、`app-x86_64-debug.apk`、`app-universal-debug.apk`

版本对应（别乱换，踩过坑）：

| 组件 | 版本 | 原因 |
|---|---|---|
| Gradle | **9.6.0** | AGP 9.5.1 要求 ≥ 9.6.0 |
| AGP | **9.5.1** | 支持 compileSdk 36 / SDK 36.1 平台 |
| JDK | **25**（Android Studio 自带 JBR） | AGP 9 需要 JDK 17+ |

---

## 7. 已修复的问题

### v2.2：改回作者原始的 imgsz 800

- 推理尺寸 640 → **800**，与数据集作者的训练设置一致
- 代价：单帧推理 ~578 ms → ~1050 ms（模拟器）
- 应用改名：包名 `com.c6h8fno4.PcbSolderDefectDetection`，显示名「PCB焊接缺陷检测」（工程被 Android Studio 重构过）
- 顺带把 `decodeFile` 加了日志，并确认过：之前的「图片解码失败」是旧 Activity 残留状态误报，不是代码问题

### v2.1：实时模式"好卡"

**现象**：进实时检测后画面像幻灯片（1.3 fps）。

**原因**有两个，叠加在一起：
1. **显示和推理串行**：每帧都要等推理（700 ms+）跑完才显示下一帧，于是预览帧率 = 推理帧率。
2. **模型太大**：YOLO26m@800，单帧推理就 700 ms–1 s。

**修复**：
1. **显示与推理解耦**：相机线程只做「转 Bitmap + 画上一次的框 + 显示」，推理放独立单线程异步执行；
   上一次没算完就跳过这次推理，但显示照常刷新。坐标因为是同一张 Bitmap，天然对齐。
2. **推理尺寸 800 → 640**：先在 687 张 val 上验证精度不掉反升（mAP50 0.9107 → 0.9151），再切换。
3. 预处理顺带从 362 ms 降到 78 ms。

**效果**：预览 1.3 → **9–19 fps**；单张推理 1036 → **578 ms**。
（期间还试过 NNAPI 硬件加速，模拟器上是参考实现、反而更慢，已改为默认关闭的常量开关。）

### v2.0：换模型为 YOLO26m（9 类焊接缺陷）

- 模型：YOLO26n（6 类裸板：缺孔/鼠咬/开路/短路/毛刺/余铜）→ **YOLO26m（9 类焊接/装配缺陷）**
- 输入尺寸 640 → **800**
- 单一阈值 → **逐类阈值**（模型卡 F1 最优点）
- ABI 去掉 `armeabi-v7a`、改成拆包（arm64 单包 108 MB，而非之前 3 ABI 合一的 90 MB）

### v1.1：退出实时检测时闪退（原生崩溃）

**现象**：进实时检测后按返回，App 直接闪退（进程 abort，不是 Java 异常）。

**崩溃栈**：`HandleUsingDestroyedMutex` → `pthread_mutex_lock` → `libonnxruntime.so` → `OrtSession.run` → `Detector.detect` ← `LiveActivity.analyze`（CameraX 分析线程）。

**原因**：`onDestroy` 直接 `detector.close()` 关了 ORT 会话，而相机线程还在 `session.run()` 里，
原生层去锁一个已销毁的 mutex → `abort()`。

**修复**：
1. `detect()` 与 `close()` 同为 `synchronized`（同一把锁）：`close()` 会等在飞那一帧跑完；`detect()` 入口检查 `closed`，关闭后返回空结果，绝不碰 session。
2. `onDestroy` 顺序：置 `destroyed` → `unbindAll()` 断相机 → `exec.shutdown()`（不用 `shutdownNow()`，避免打断原生调用）→ 关会话放**后台线程**（不能在主线程等）。
3. UI 回调检查 `destroyed`；先 `setImageDrawable(null)` 再 `recycle()`。

**验证**：4 轮「进实时→推理中返回」+ 5 轮「单张检测中直接退出」，crash buffer 全空、无 ANR。

---


### v2.3：实时模式"基本检测不到任何东西"

**现象**：实时检测画面在动，但几乎永远检出 0 个缺陷。

**根因（实测定位，不是猜）**：

```
java.lang.IllegalStateException: Immutable bitmap passed to Canvas constructor
    at LiveActivity.processFrame(LiveActivity.java:199)
```

`processFrame` 里对相机帧调 `new Canvas(frame)` 画框，但某些来源的 Bitmap 是**不可变**的
（旋转、`copy(..., false)` 产生），`new Canvas` 直接抛异常 → 处理线程被打死 →
之后再也不处理任何帧 → 表现为"画面在动但永远检测不到"。

**三处修复**：

1. 画框前检查 `frame.isMutable()`，不可变就用 `copy(..., true)` 转一份可变的；并把整个
   `processFrame` 包进 try/catch，任何异常都不许打死处理线程（避免同类问题再次静默失败）。
2. **分析分辨率提到 1920×1080**（原来默认约 640×480）：焊点缺陷很小，低分辨率帧缩到 800 后
   细节全丢，这是"检不到"的第二个原因。实测帧从 640×480 变成 960×1280。
3. 实时模式置信度改为**逐类阈值 × 0.6**（`LIVE_CONF_SCALE`）：小缺陷分数普遍偏低，用缩放系数
   换召回。要更灵敏就调小，要更少误报就调大。

**新增诊断信息**（HUD 第二行）：`缺陷 N 个 · 最高分 0.xx`。
最高分是**全图不看阈值的最高类别分**，能直接区分两种情况：

- 最高分 0.03 → 画面里确实没有模型认识的东西（或角度/光照不对）
- 最高分 0.4 但缺陷 0 → 分数没问题，是阈值/后处理把框滤掉了

**验证**：用固定图片模式（`--es still_image <路径>`）把真实焊点图灌进完整实时管线，
帧 1280×1280、检出 2 个 `excess_solder`、最高分 0.81、框正确落在焊盘上；相机模式帧 960×1280、
对着非 PCB 场景最高分 0.03、零误报、无崩溃。
---

## 8. 已知限制

- **实时模式的「框」刷新只有 2–3 次/秒**（真机估），预览本身是流畅的。这是 20M 参数模型的成本，要更快得 int8 量化或换小模型。
- **APK 108 MB**，主要来自 78 MB 模型；int8 量化可把模型压到 ~20 MB 且通常更快，但要重新验证精度（尤其 `solder_void` ↔ `insufficient_solder` 本来就容易混）。
- **NNAPI 默认关闭**（`Detector.USE_NNAPI = false`）。真机上想试就改 true 重编，失败会自动回退 CPU；部分设备 NNAPI 反而更慢。
- **模型卡的已知弱点**：`solder_void` 常被误判成 `insufficient_solder`（约 71% 的空洞错误落在这）；`burn` 召回偏低（~62%）；某个子集的暗光照片上表现下降。
- 实时模式显示的是被检测的那一帧（800 长边 letterbox 后），不是高分辨率预览。
- 单张模式不保留历史、不导出 CSV。
- **许可**：模型卡标注 `other / research-use-mixed-sources`，数据集许可要求商用前联系作者；Ultralytics 本身是 AGPL-3.0。对外分发前请确认。

---

## 9. 调试钩子

```bash
adb push test.jpg /data/local/tmp/test.jpg
adb shell run-as com.pcb.defect cp /data/local/tmp/test.jpg /data/data/com.pcb.defect/files/t.jpg
adb shell am start -n com.pcb.defect/.MainActivity \
    --es image_path /data/data/com.pcb.defect/files/t.jpg
```

路径必须在 App 可读的位置（私有目录）；`/sdcard/Download/...` 会因分区存储读不到——正常使用请走「从相册选图」。