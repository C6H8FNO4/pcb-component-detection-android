package com.c6h8fno4.PcbSolderDefectDetection;

import android.content.Context;
import android.graphics.Bitmap;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

/**
 * 本地 ONNX 检测器（纯离线）。
 *
 * 当前模型：YOLO26m · PCB 焊接/装配缺陷 · 9 类 · 训练 imgsz=800
 *   参数 21,786,766 / ONNX fp32 约 87 MB
 *
 * 模型输出 (1, 13, N)：前 4 行 cx,cy,w,h（800 输入坐标系的像素值），
 * 后 9 行是 9 个类别的分数（已 sigmoid，无 objectness）→ 需自行解码 + NMS。
 *
 * 置信度阈值来自模型卡（val 上 F1 最优点，逐类不同），见 CONF_PER_CLASS。
 */
public class Detector implements AutoCloseable {

    /** 训练时的输入边长（letterbox 目标尺寸） */
    public static final int INPUT_SIZE = 800;
    public static final int NUM_CLASSES = 9;

    public static final String[] LABELS = {
            "insufficient_solder", "excess_solder", "solder_bridge", "solder_void",
            "missing_component", "shifted_component", "burn", "physical_damage", "oxidation"
    };
    public static final String[] LABELS_CN = {
            "少锡", "多锡", "连锡", "空洞",
            "缺件", "偏移", "烧焦", "损伤", "氧化"
    };
    /** 逐类 F1 最优阈值（模型卡给定） */
    public static final float[] CONF_PER_CLASS = {
            0.297f, 0.349f, 0.417f, 0.249f, 0.381f, 0.389f, 0.224f, 0.214f, 0.228f
    };
    /** 全部类别的兜底阈值 */
    public static final float CONF_GLOBAL = 0.228f;

    /** 单张最多保留的候选框，防止低阈值时 NMS 爆炸 */
    private static final int MAX_CANDIDATES = 300;

    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;
    /** 关闭标志：与 detect() 共用同一把锁，避免"关闭后仍有推理在跑"导致原生崩溃 */
    private boolean closed = false;

    private final float[] inputBuf = new float[3 * INPUT_SIZE * INPUT_SIZE];
    private int[] pixelBuf = new int[INPUT_SIZE * INPUT_SIZE];

    /**
     * 是否尝试 NNAPI 硬件加速。
     *
     * 默认关闭：模拟器上 NNAPI 是「参考实现」（用 CPU 模拟 DSP），实测比纯 CPU 慢一个量级；
     * 部分真机的 NNAPI 也比 CPU 慢或结果不稳定。
     * 如果你在真机上想试，把它改成 true 重新编译即可（失败会自动回退 CPU）。
     */
    private static final boolean USE_NNAPI = false;

    public Detector(Context ctx, String assetModel) throws Exception {
        env = OrtEnvironment.getEnvironment();
        byte[] model = readAsset(ctx, assetModel);
        int threads = Math.max(2, Runtime.getRuntime().availableProcessors() / 2);
        session = createSession(model, threads, USE_NNAPI);

        Iterator<String> it = session.getInputNames().iterator();
        inputName = it.hasNext() ? it.next() : "images";
    }

    /**
     * 建会话。先尝试 NNAPI（真机上可走 DSP/NPU，通常快 2 倍以上），
     * 失败或设备不支持就自动回退纯 CPU（XNNPACK）。
     */
    private OrtSession createSession(byte[] model, int threads, boolean tryNnapi) throws Exception {
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        opts.setIntraOpNumThreads(threads);
        if (tryNnapi) {
            try {
                opts.addNnapi();
            } catch (Throwable ignored) {
                // 老设备或非 NNAPI 版 ORT：忽略，用 CPU
            }
        }
        try {
            return env.createSession(model, opts);
        } catch (Exception e) {
            if (!tryNnapi) throw e;
            // NNAPI 建会话失败 → 回退 CPU 再试一次
            return createSession(model, threads, false);
        }
    }

    public static class Box {
        public int cls;
        public float conf;
        public float x1, y1, x2, y2;

        public String label() {
            return LABELS[cls];
        }

        public String labelCn() {
            return LABELS_CN[cls];
        }
    }

    public static class Result {
        public final List<Box> boxes = new ArrayList<>();
        public long preMs, inferMs, postMs;
        public int imgW, imgH;
        /** 全图最高类别分数（不看阈值），用于诊断为什么没检出 */
        public float maxScore;

        public int countByClass(int cls) {
            int n = 0;
            for (Box b : boxes) if (b.cls == cls) n++;
            return n;
        }
    }

    /** 用逐类阈值检测（推荐，与模型卡一致） */
    public synchronized Result detect(Bitmap src, float iouThres) {
        return detect(src, -1f, iouThres, true, 1f);
    }

    /** 逐类阈值 + 缩放系数：实时模式用 <1 的系数换召回（小缺陷分数普遍偏低） */
    public synchronized Result detectScaled(Bitmap src, float iouThres, float confScale) {
        return detect(src, -1f, iouThres, true, confScale);
    }

    /** 用统一阈值检测（阈值不宜低于 0.228） */
    public synchronized Result detect(Bitmap src, float confThres, float iouThres) {
        return detect(src, confThres, iouThres, false, 1f);
    }

    public synchronized Result detect(Bitmap src, float confThres, float iouThres, boolean perClass) {
        return detect(src, confThres, iouThres, perClass, 1f);
    }

    public synchronized Result detect(Bitmap src, float confThres, float iouThres,
                                    boolean perClass, float confScale) {
        Result r = new Result();
        r.imgW = src.getWidth();
        r.imgH = src.getHeight();
        // 已经 close() 过就直接返回空结果，绝不能再碰 session
        if (closed) return r;

        long t0 = System.currentTimeMillis();

        // ---------- 1. letterbox 到 INPUT_SIZE ----------
        final int S = INPUT_SIZE;
        float scale = Math.min(S / (float) r.imgW, S / (float) r.imgH);
        int nw = Math.max(1, Math.round(r.imgW * scale));
        int nh = Math.max(1, Math.round(r.imgH * scale));
        int padX = (S - nw) / 2;
        int padY = (S - nh) / 2;

        Bitmap scaled = Bitmap.createScaledBitmap(src, nw, nh, true);
        final int n = nw * nh;
        if (pixelBuf.length < n) pixelBuf = new int[n];
        scaled.getPixels(pixelBuf, 0, nw, 0, 0, nw, nh);
        if (scaled != src) scaled.recycle();

        Arrays.fill(inputBuf, 114f / 255f);
        final int plane = S * S;
        for (int y = 0; y < nh; y++) {
            int rowSrc = y * nw;
            int rowDst = (y + padY) * S + padX;
            for (int x = 0; x < nw; x++) {
                int c = pixelBuf[rowSrc + x];
                int idx = rowDst + x;
                inputBuf[idx] = ((c >> 16) & 0xFF) / 255f;        // R
                inputBuf[plane + idx] = ((c >> 8) & 0xFF) / 255f;  // G
                inputBuf[2 * plane + idx] = (c & 0xFF) / 255f;     // B
            }
        }
        r.preMs = System.currentTimeMillis() - t0;

        // ---------- 2. 推理 ----------
        long t1 = System.currentTimeMillis();
        float[] out;
        int channels, anchors;
        long[] shape = {1, 3, S, S};
        try (OnnxTensor in = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputBuf), shape);
             OrtSession.Result res = session.run(Collections.singletonMap(inputName, in))) {
            OnnxTensor o = (OnnxTensor) res.get(0);
            long[] os = o.getInfo().getShape();     // [1, 4+nc, anchors]
            channels = (int) os[1];
            anchors = (int) os[2];
            out = new float[channels * anchors];
            o.getFloatBuffer().get(out);
        } catch (Exception e) {
            throw new RuntimeException("推理失败: " + e.getMessage(), e);
        }
        r.inferMs = System.currentTimeMillis() - t1;

        // ---------- 3. 解码 + 逐类 NMS ----------
        long t2 = System.currentTimeMillis();
        int numCls = Math.min(NUM_CLASSES, channels - 4);
        for (int c = 0; c < numCls; c++) {
            float th = (perClass ? CONF_PER_CLASS[c] : confThres) * confScale;
            int scoreRow = (4 + c) * anchors;
            List<Box> cand = new ArrayList<>();
            for (int i = 0; i < anchors; i++) {
                float s = out[scoreRow + i];
                if (s > r.maxScore) r.maxScore = s;
                if (s < th) continue;
                float cx = out[i];
                float cy = out[anchors + i];
                float w = out[2 * anchors + i];
                float h = out[3 * anchors + i];

                Box b = new Box();
                b.cls = c;
                b.conf = s;
                b.x1 = clamp((cx - w / 2f - padX) / scale, 0, r.imgW - 1);
                b.y1 = clamp((cy - h / 2f - padY) / scale, 0, r.imgH - 1);
                b.x2 = clamp((cx + w / 2f - padX) / scale, 0, r.imgW - 1);
                b.y2 = clamp((cy + h / 2f - padY) / scale, 0, r.imgH - 1);
                if (b.x2 - b.x1 < 1 || b.y2 - b.y1 < 1) continue;
                cand.add(b);
            }
            if (cand.isEmpty()) continue;

            cand.sort((a, b) -> Float.compare(b.conf, a.conf));
            if (cand.size() > MAX_CANDIDATES) cand = cand.subList(0, MAX_CANDIDATES);

            boolean[] dead = new boolean[cand.size()];
            for (int i = 0; i < cand.size(); i++) {
                if (dead[i]) continue;
                Box bi = cand.get(i);
                r.boxes.add(bi);
                for (int j = i + 1; j < cand.size(); j++) {
                    if (!dead[j] && iou(bi, cand.get(j)) > iouThres) dead[j] = true;
                }
            }
        }
        r.boxes.sort((a, b) -> Float.compare(b.conf, a.conf));
        r.postMs = System.currentTimeMillis() - t2;
        return r;
    }

    /**
     * 释放模型与 ONNX Runtime 会话。
     *
     * 与 detect() 同为 synchronized：如果此刻还有一帧在推理，close() 会等它跑完
     * 再释放，避免 ONNX Runtime 原生层出现 "使用已销毁的锁" 崩溃。
     */
    @Override
    public synchronized void close() {
        if (closed) return;
        closed = true;
        try {
            if (session != null) session.close();
        } catch (Exception ignored) {
        }
    }

    // ---------------------------------------------------------------- 工具

    private static float clamp(float v, float lo, float hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    private static float iou(Box a, Box b) {
        float ix1 = Math.max(a.x1, b.x1);
        float iy1 = Math.max(a.y1, b.y1);
        float ix2 = Math.min(a.x2, b.x2);
        float iy2 = Math.min(a.y2, b.y2);
        float iw = ix2 - ix1;
        float ih = iy2 - iy1;
        if (iw <= 0 || ih <= 0) return 0f;
        float inter = iw * ih;
        float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
        float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
        return inter / (areaA + areaB - inter);
    }

    private static byte[] readAsset(Context ctx, String name) throws Exception {
        try (InputStream is = ctx.getAssets().open(name);
             ByteArrayOutputStream bos = new ByteArrayOutputStream(100 * 1024 * 1024)) {
            byte[] tmp = new byte[64 * 1024];
            int len;
            while ((len = is.read(tmp)) > 0) bos.write(tmp, 0, len);
            return bos.toByteArray();
        }
    }
}