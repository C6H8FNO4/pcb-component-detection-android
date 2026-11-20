package com.c6h8fno4.PcbSolderDefectDetection;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.core.content.FileProvider;

import java.io.File;
import java.io.InputStream;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * 单张检测界面：相册选图 / 拍照 → 本地推理 → 框 + 明细。
 * 实时检测入口在此跳转 LiveActivity。
 */
public class MainActivity extends Activity {

    private static final int REQ_PICK = 1001;
    private static final int REQ_PHOTO = 1002;
    private static final int REQ_PERM_CAMERA = 1003;
    private static final int REQ_LIVE = 1004;

    /** NMS 的 IoU 阈值；置信度用模型卡给的逐类阈值（Detector.CONF_PER_CLASS） */
    private static final float IOU = 0.45f;

    private ImageView preview;
    private TextView status, summary, detail;
    private Button btnPick, btnPhoto, btnLive;

    private volatile Detector detector;
    private final ExecutorService io = Executors.newSingleThreadExecutor();
    private final Handler ui = new Handler(Looper.getMainLooper());

    private Uri pendingPhotoUri;
    private Bitmap lastResultBitmap;
    /** 调试/自动化入口：am start ... --es image_path /sdcard/x.jpg */
    private String pendingImagePath;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        preview = findViewById(R.id.preview);
        status = findViewById(R.id.status);
        summary = findViewById(R.id.summary);
        detail = findViewById(R.id.detail);
        btnPick = findViewById(R.id.btnPick);
        btnPhoto = findViewById(R.id.btnPhoto);
        btnLive = findViewById(R.id.btnLive);

        btnPick.setEnabled(false);
        btnPhoto.setEnabled(false);
        btnLive.setEnabled(false);

        btnPick.setOnClickListener(v -> pickImage());
        btnPhoto.setOnClickListener(v -> takePhoto());
        btnLive.setOnClickListener(v -> openLive());

        if (getIntent() != null) {
            pendingImagePath = getIntent().getStringExtra("image_path");
            // 测试用：把 still_image 转发给实时页，用固定图片跑完整实时管线
            String still = getIntent().getStringExtra("still_image");
            if (still != null) {
                Intent i = new Intent(this, LiveActivity.class);
                i.putExtra("still_image", still);
                startActivity(i);
                finish();
                return;
            }
        }

        loadModelAsync();
    }

    // ------------------------------------------------------------ 模型加载

    private void loadModelAsync() {
        status.setText(R.string.loading_model);
        io.execute(() -> {
            try {
                long t0 = System.currentTimeMillis();
                Detector d = new Detector(getApplicationContext(), "best.onnx");
                long dt = System.currentTimeMillis() - t0;
                ui.post(() -> {
                    detector = d;
                    status.setText(getString(R.string.model_ready) + " · 加载 " + dt + " ms");
                    btnPick.setEnabled(true);
                    btnPhoto.setEnabled(true);
                    btnLive.setEnabled(true);
                    if (pendingImagePath != null && !pendingImagePath.isEmpty()) {
                        runDetectOnPath(pendingImagePath);
                        pendingImagePath = null;
                    }
                });
            } catch (Throwable e) {
                ui.post(() -> status.setText("模型加载失败：" + e.getMessage()));
            }
        });
    }

    // ------------------------------------------------------------ 取图

    private void pickImage() {
        Intent i = new Intent(Intent.ACTION_GET_CONTENT);
        i.setType("image/*");
        i.addCategory(Intent.CATEGORY_OPENABLE);
        startActivityForResult(Intent.createChooser(i, getString(R.string.pick_image)), REQ_PICK);
    }

    private void takePhoto() {
        try {
            File dir = new File(getCacheDir(), "captures");
            if (!dir.exists()) dir.mkdirs();
            File f = new File(dir, "shot_" + System.currentTimeMillis() + ".jpg");
            pendingPhotoUri = FileProvider.getUriForFile(
                    this, getPackageName() + ".fileprovider", f);

            Intent i = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            i.putExtra(MediaStore.EXTRA_OUTPUT, pendingPhotoUri);
            i.addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
            startActivityForResult(i, REQ_PHOTO);
        } catch (Exception e) {
            toast("打开相机失败：" + e.getMessage());
        }
    }

    private void openLive() {
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, REQ_PERM_CAMERA);
            return;
        }
        startActivityForResult(new Intent(this, LiveActivity.class), REQ_LIVE);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] results) {
        super.onRequestPermissionsResult(requestCode, permissions, results);
        if (requestCode == REQ_PERM_CAMERA) {
            if (results.length > 0 && results[0] == PackageManager.PERMISSION_GRANTED) {
                startActivityForResult(new Intent(this, LiveActivity.class), REQ_LIVE);
            } else {
                toast(getString(R.string.camera_permission_needed));
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_OK) return;

        if (requestCode == REQ_PICK && data != null && data.getData() != null) {
            runDetectOnUri(data.getData());
        } else if (requestCode == REQ_PHOTO && pendingPhotoUri != null) {
            runDetectOnUri(pendingPhotoUri);
        }
    }

    // ------------------------------------------------------------ 检测

    private void runDetectOnUri(Uri uri) {
        if (detector == null) {
            toast("模型还没准备好");
            return;
        }
        summary.setText(R.string.detecting);
        detail.setText("");
        io.execute(() -> {
            try {
                Bitmap bmp = decode(uri);
                if (bmp == null) {
                    ui.post(() -> summary.setText("图片解码失败"));
                    return;
                }
                Detector.Result r = detector.detect(bmp, IOU);
                Bitmap drawn = BoxRenderer.draw(bmp, r);
                ui.post(() -> showResult(drawn, r, bmp));
            } catch (Throwable e) {
                ui.post(() -> summary.setText("检测出错：" + e.getMessage()));
            }
        });
    }

    /** 直接对文件路径检测（调试 / adb 自动化用） */
    private void runDetectOnPath(String path) {
        if (detector == null) {
            toast("模型还没准备好");
            return;
        }
        summary.setText(R.string.detecting);
        detail.setText("");
        io.execute(() -> {
            try {
                Bitmap bmp = decodeFile(path);
                if (bmp == null) {
                    ui.post(() -> summary.setText("图片解码失败：" + path));
                    return;
                }
                Detector.Result r = detector.detect(bmp, IOU);
                Bitmap drawn = BoxRenderer.draw(bmp, r);
                ui.post(() -> showResult(drawn, r, bmp));
            } catch (Throwable e) {
                ui.post(() -> summary.setText("检测出错：" + e.getMessage()));
            }
        });
    }

    private Bitmap decodeFile(String path) {
        java.io.File f = new java.io.File(path);
        android.util.Log.i("PCB", "decodeFile: path=" + path + " exists=" + f.exists()
                + " readable=" + f.canRead() + " len=" + f.length());
        BitmapFactory.Options b = new BitmapFactory.Options();
        b.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(path, b);
        android.util.Log.i("PCB", "bounds=" + b.outWidth + "x" + b.outHeight
                + " mime=" + b.outMimeType);
        if (b.outWidth <= 0 || b.outHeight <= 0) {
            android.util.Log.e("PCB", "decodeFile: bounds 解析失败，无法解码该文件");
            return null;
        }
        int max = Math.max(b.outWidth, b.outHeight);
        int sample = 1;
        while (max / sample > 2000) sample *= 2;

        BitmapFactory.Options o = new BitmapFactory.Options();
        o.inSampleSize = sample;
        o.inPreferredConfig = Bitmap.Config.ARGB_8888;
        Bitmap out = BitmapFactory.decodeFile(path, o);
        android.util.Log.i("PCB", "decodeFile result=" + (out == null ? "null" : out.getWidth() + "x" + out.getHeight()));
        return out;
    }

    private void showResult(Bitmap drawn, Detector.Result r, Bitmap original) {
        if (lastResultBitmap != null) lastResultBitmap.recycle();
        lastResultBitmap = drawn;
        preview.setImageBitmap(drawn);

        if (r.boxes.isEmpty()) {
            summary.setText(getString(R.string.no_defect) + "   (逐类阈值)");
        } else {
            StringBuilder sb = new StringBuilder();
            sb.append("共 ").append(r.boxes.size()).append(" 个缺陷：");
            for (int c = 0; c < Detector.NUM_CLASSES; c++) {
                int n = r.countByClass(c);
                if (n > 0) sb.append("  ").append(Detector.LABELS_CN[c]).append("×").append(n);
            }
            sb.append("\n耗时：预处理 ").append(r.preMs)
                    .append(" ms · 推理 ").append(r.inferMs)
                    .append(" ms · 后处理 ").append(r.postMs).append(" ms");
            summary.setText(sb.toString());

            StringBuilder d = new StringBuilder();
            int i = 1;
            for (Detector.Box b : r.boxes) {
                d.append(String.format(Locale.US, "%2d. %-15s %.3f  (%d,%d)-(%d,%d)\n",
                        i++, b.label(), b.conf,
                        (int) b.x1, (int) b.y1, (int) b.x2, (int) b.y2));
            }
            detail.setText(d.toString());
        }
        original.recycle();
    }

    /** 按需降采样解码，避免大图 OOM，同时保留足够细节 */
    private Bitmap decode(Uri uri) throws Exception {
        BitmapFactory.Options bounds = new BitmapFactory.Options();
        bounds.inJustDecodeBounds = true;
        try (InputStream is = getContentResolver().openInputStream(uri)) {
            BitmapFactory.decodeStream(is, null, bounds);
        }
        int max = Math.max(bounds.outWidth, bounds.outHeight);
        int sample = 1;
        while (max / sample > 2000) sample *= 2;

        BitmapFactory.Options opts = new BitmapFactory.Options();
        opts.inSampleSize = sample;
        opts.inPreferredConfig = Bitmap.Config.ARGB_8888;
        try (InputStream is = getContentResolver().openInputStream(uri)) {
            return BitmapFactory.decodeStream(is, null, opts);
        }
    }

    private void toast(String msg) {
        Toast.makeText(this, msg, Toast.LENGTH_SHORT).show();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        // 不打断正在跑的推理：shutdown() 让在飞的任务自然结束，
        // 关闭 ORT 会话放到后台线程（close() 与 detect() 同锁，会等它跑完）。
        if (io != null) io.shutdown();
        Detector d = detector;
        detector = null;
        if (d != null) new Thread(d::close, "det-close").start();

        // 先解绑 ImageView 再回收，避免窗口还持有已回收的 Bitmap
        if (preview != null) preview.setImageDrawable(null);
        if (lastResultBitmap != null) {
            lastResultBitmap.recycle();
            lastResultBitmap = null;
        }
    }
}