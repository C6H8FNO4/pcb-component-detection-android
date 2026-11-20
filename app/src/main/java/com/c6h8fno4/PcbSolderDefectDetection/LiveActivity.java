package com.c6h8fno4.PcbSolderDefectDetection;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Size;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.resolutionselector.ResolutionSelector;
import androidx.camera.core.resolutionselector.ResolutionStrategy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.app.ComponentActivity;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * 实时检测。
 *
 * v2.3 关键改动：
 *   1) 分析分辨率提到 1920×1080 —— 焊点缺陷很小，640×480 的相机帧缩到 800 后细节全丢，
 *      这是"实时检不到东西"的主因。
 *   2) 实时模式置信度用逐类阈值 × LIVE_CONF_SCALE（默认 0.6）—— 换召回。
 *   3) HUD 显示帧尺寸和"全图最高分"，没检出时能看到离阈值还有多远。
 *   4) 支持 --es still_image <path> 用固定图片跑完整实时管线（验证用，不需要相机）。
 *
 * 显示与推理解耦：每帧立刻显示并画上一次的框；推理在独立线程异步跑。
 */
public class LiveActivity extends ComponentActivity {

    private static final float IOU = 0.45f;
    /** 实时模式置信度缩放：逐类阈值 × 该系数。调小更容易检出、误报也会变多 */
    private static final float LIVE_CONF_SCALE = 0.6f;

    private ImageView frameView;
    private TextView hud;

    private volatile Detector detector;
    private ExecutorService analyzerExec;
    private ExecutorService inferExec;

    private final Handler ui = new Handler(Looper.getMainLooper());
    private final AtomicBoolean inferring = new AtomicBoolean(false);
    private volatile boolean destroyed = false;

    private ProcessCameraProvider cameraProvider;
    private Bitmap shownBitmap;

    private volatile List<Detector.Box> lastBoxes = new ArrayList<>();
    private volatile int boxesW = 0, boxesH = 0;
    private volatile long lastInferMs = 0;
    private volatile float lastMaxScore = 0f;
    private volatile String frameInfo = "";

    private long lastFrameTs = 0;
    private float previewFps = 0f;
    private final Paint boxPaint = new Paint(Paint.ANTI_ALIAS_FLAG);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_live);

        frameView = findViewById(R.id.frame);
        hud = findViewById(R.id.hud);
        findViewById(R.id.btnBack).setOnClickListener(v -> finish());

        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(4f);

        hud.setText("正在加载模型…");
        analyzerExec = Executors.newSingleThreadExecutor();
        inferExec = Executors.newSingleThreadExecutor();

        analyzerExec.execute(() -> {
            try {
                Detector d = new Detector(getApplicationContext(), "best.onnx");
                ui.post(() -> {
                    if (destroyed) {
                        new Thread(d::close, "det-close").start();
                        return;
                    }
                    detector = d;
                    String still = getIntent() != null ? getIntent().getStringExtra("still_image") : null;
                    if (still != null) {
                        hud.setText("固定图片测试模式");
                        startStillLoop(still);
                    } else {
                        hud.setText("对准 PCB 板");
                        startCamera();
                    }
                });
            } catch (Throwable e) {
                ui.post(() -> hud.setText("模型加载失败: " + e.getMessage()));
            }
        });
    }

    // ---------------------------------------------------------------- 相机

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> future = ProcessCameraProvider.getInstance(this);
        future.addListener(() -> {
            if (destroyed) return;
            try {
                cameraProvider = future.get();
                bindAnalysis();
            } catch (Exception e) {
                hud.setText("相机启动失败: " + e.getMessage());
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void bindAnalysis() {
        // 尽量给高分辨率：小缺陷经不起下采样
        ResolutionSelector selector = new ResolutionSelector.Builder()
                .setResolutionStrategy(new ResolutionStrategy(
                        new Size(1920, 1080),
                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
                .build();

        ImageAnalysis analysis = new ImageAnalysis.Builder()
                .setResolutionSelector(selector)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build();

        analysis.setAnalyzer(analyzerExec, this::analyze);

        cameraProvider.unbindAll();
        cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, analysis);
    }

    /** 每帧回调：转换 → 画旧框 → 显示；推理异步 */
    private void analyze(@NonNull ImageProxy image) {
        if (destroyed) {
            image.close();
            return;
        }
        Bitmap frame = null;
        try {
            frame = toBitmap(image);
        } finally {
            image.close();
        }
        if (frame == null) return;
        processFrame(frame);
    }

    /** 固定图片模式：把同一张图反复喂进同一条管线（无需相机，便于验证/演示） */
    private void startStillLoop(String path) {
        analyzerExec.execute(() -> {
            Bitmap src = BitmapFactory.decodeFile(path);
            if (src == null) {
                ui.post(() -> hud.setText("固定图片读取失败: " + path));
                return;
            }
            while (!destroyed) {
                Bitmap copy = src.copy(Bitmap.Config.ARGB_8888, true);   // 必须可变：后面要往上画框
                if (copy != null) processFrame(copy);
                try {
                    Thread.sleep(250);
                } catch (InterruptedException e) {
                    break;
                }
            }
            src.recycle();
        });
    }

    /** 显示 + 异步推理（相机和固定图片共用） */
    private void processFrame(Bitmap frame) {
        if (destroyed) {
            frame.recycle();
            return;
        }
        try {
            // 画框要求可变位图；某些来源（旋转、copy 出来的）可能不可变，
            // 不可变时 new Canvas() 会抛 IllegalStateException 直接打死线程。
            if (!frame.isMutable()) {
                Bitmap mutable = frame.copy(Bitmap.Config.ARGB_8888, true);
                frame.recycle();
                if (mutable == null) return;
                frame = mutable;
            }

            List<Detector.Box> boxes = lastBoxes;
            if (!boxes.isEmpty()) {
                Canvas c = new Canvas(frame);
                drawBoxes(c, boxes, boxesW, boxesH, frame.getWidth(), frame.getHeight());
            }

            maybeInfer(frame);

            long now = System.currentTimeMillis();
            if (lastFrameTs > 0) {
                float inst = 1000f / Math.max(1, now - lastFrameTs);
                previewFps = previewFps == 0 ? inst : previewFps * 0.85f + inst * 0.15f;
            }
            lastFrameTs = now;

            final String text = String.format(Locale.US,
                    "预览 %.0f fps · 推理 %d ms · 帧 %s\n缺陷 %d 个 · 最高分 %.2f\n%s",
                    previewFps, lastInferMs, frameInfo, boxes.size(), lastMaxScore, brief(boxes));
            final Bitmap toShow = frame;

            ui.post(() -> {
                if (destroyed) {
                    toShow.recycle();
                    return;
                }
                Bitmap old = shownBitmap;
                shownBitmap = toShow;
                frameView.setImageBitmap(toShow);
                hud.setText(text);
                if (old != null && old != toShow) old.recycle();
            });
        } catch (Throwable e) {
            // 任何异常都不许打死处理线程，否则表现为“再也检测不到东西”
            frame.recycle();
            ui.post(() -> {
                if (!destroyed) hud.setText("处理异常: " + e);
            });
        }
    }

    private void maybeInfer(Bitmap frame) {
        if (!inferring.compareAndSet(false, true)) return;
        final Detector local = detector;
        if (local == null) {
            inferring.set(false);
            return;
        }
        final Bitmap copy;
        try {
            copy = frame.copy(Bitmap.Config.ARGB_8888, false);
        } catch (Throwable t) {
            inferring.set(false);
            return;
        }
        final int w = frame.getWidth(), h = frame.getHeight();

        inferExec.execute(() -> {
            try {
                if (destroyed) return;
                // 实时模式：逐类阈值 × LIVE_CONF_SCALE，换召回来对抗小缺陷分数偏低
                Detector.Result r = local.detectScaled(copy, IOU, LIVE_CONF_SCALE);
                lastBoxes = r.boxes;
                boxesW = w;
                boxesH = h;
                lastInferMs = r.inferMs;
                lastMaxScore = r.maxScore;
                frameInfo = w + "x" + h;
            } catch (Throwable ignored) {
            } finally {
                copy.recycle();
                inferring.set(false);
            }
        });
    }

    private void drawBoxes(Canvas c, List<Detector.Box> boxes, int srcW, int srcH, int dstW, int dstH) {
        if (srcW <= 0 || srcH <= 0) return;
        float sx = dstW / (float) srcW;
        float sy = dstH / (float) srcH;
        RectF r = new RectF();
        for (Detector.Box b : boxes) {
            boxPaint.setColor(BoxRenderer.colorOf(b.cls));
            r.set(b.x1 * sx, b.y1 * sy, b.x2 * sx, b.y2 * sy);
            c.drawRect(r, boxPaint);
        }
    }

    private String brief(List<Detector.Box> boxes) {
        if (boxes.isEmpty()) return "—";
        int[] counts = new int[Detector.NUM_CLASSES];
        for (Detector.Box b : boxes) counts[b.cls]++;
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < counts.length; i++) {
            if (counts[i] > 0) sb.append(Detector.LABELS_CN[i]).append("×").append(counts[i]).append("  ");
        }
        return sb.toString().trim();
    }

    /** ImageProxy(RGBA_8888) → Bitmap，并按 sensor 方向旋转正 */
    private Bitmap toBitmap(ImageProxy image) {
        ImageProxy.PlaneProxy plane = image.getPlanes()[0];
        ByteBuffer buf = plane.getBuffer();
        int rowStride = plane.getRowStride();
        int pixelStride = plane.getPixelStride();
        int bufW = rowStride / pixelStride;
        int h = image.getHeight();

        Bitmap raw;
        try {
            raw = Bitmap.createBitmap(bufW, h, Bitmap.Config.ARGB_8888);
            raw.copyPixelsFromBuffer(buf);
        } catch (Throwable t) {
            return null;
        }

        Bitmap cropped = raw;
        if (bufW != image.getWidth()) {
            cropped = Bitmap.createBitmap(raw, 0, 0, image.getWidth(), h);
            raw.recycle();
        }

        int deg = image.getImageInfo().getRotationDegrees();
        if (deg % 360 == 0) return cropped;

        Matrix m = new Matrix();
        m.postRotate(deg);
        Bitmap rotated = Bitmap.createBitmap(cropped, 0, 0,
                cropped.getWidth(), cropped.getHeight(), m, true);
        if (rotated != cropped) cropped.recycle();
        return rotated;
    }

    @Override
    protected void onDestroy() {
        destroyed = true;

        if (cameraProvider != null) {
            try {
                cameraProvider.unbindAll();
            } catch (Throwable ignored) {
            }
        }
        if (analyzerExec != null) analyzerExec.shutdown();
        if (inferExec != null) inferExec.shutdown();

        Detector d = detector;
        detector = null;
        if (d != null) new Thread(d::close, "det-close").start();

        if (frameView != null) frameView.setImageDrawable(null);
        if (shownBitmap != null) {
            shownBitmap.recycle();
            shownBitmap = null;
        }
        lastBoxes = new ArrayList<>();

        super.onDestroy();
    }
}