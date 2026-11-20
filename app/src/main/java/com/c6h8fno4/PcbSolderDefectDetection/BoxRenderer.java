package com.c6h8fno4.PcbSolderDefectDetection;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;

/** 把检测结果画到图上（单张模式和实时模式共用）。 */
public final class BoxRenderer {

    /** 每个类别一个颜色，方便肉眼区分（9 类） */
    private static final int[] COLORS = {
            0xFFFF5252, // insufficient_solder 少锡 红
            0xFFFFB300, // excess_solder 多锡 橙
            0xFF40C4FF, // solder_bridge 连锡 蓝
            0xFFFF4081, // solder_void 空洞 粉
            0xFF7C4DFF, // missing_component 缺件 紫
            0xFF00E676, // shifted_component 偏移 绿
            0xFFFF6E40, // burn 烧焦 深橙
            0xFFE040FB, // physical_damage 损伤 洋红
            0xFF18FFFF  // oxidation 氧化 青
    };

    private BoxRenderer() {
    }

    public static int colorOf(int cls) {
        return COLORS[cls % COLORS.length];
    }

    /** 返回新的已标注 Bitmap（不修改原图）。 */
    public static Bitmap draw(Bitmap src, Detector.Result r) {
        Bitmap out = src.copy(Bitmap.Config.ARGB_8888, true);
        Canvas c = new Canvas(out);
        drawOn(c, r, out.getWidth(), out.getHeight());
        return out;
    }

    /** 在已有 Canvas 上画框；w/h 是画布尺寸，用来按比例决定线宽和字号。 */
    public static void drawOn(Canvas c, Detector.Result r, int w, int h) {
        if (r == null || r.boxes.isEmpty()) return;

        float base = Math.max(w, h) / 640f;
        float stroke = Math.max(2f, 3f * base);
        float textSize = Math.max(11f, 16f * base);

        Paint boxPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(stroke);

        Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        textPaint.setTextSize(textSize);
        textPaint.setFakeBoldText(true);
        textPaint.setColor(Color.WHITE);

        Paint bgPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        bgPaint.setStyle(Paint.Style.FILL);

        RectF rect = new RectF();
        for (Detector.Box b : r.boxes) {
            int color = colorOf(b.cls);
            boxPaint.setColor(color);
            rect.set(b.x1, b.y1, b.x2, b.y2);
            c.drawRect(rect, boxPaint);

            String text = b.label() + " " + String.format("%.2f", b.conf);
            float tw = textPaint.measureText(text);
            float th = textSize * 1.15f;

            float tx = b.x1;
            float ty = b.y1 - th - stroke;
            if (ty < 0) ty = b.y1 + stroke;      // 顶部放不下就画在框内
            if (tx + tw > w) tx = Math.max(0, w - tw);

            bgPaint.setColor(color);
            c.drawRect(tx, ty, tx + tw + textSize * 0.4f, ty + th, bgPaint);
            c.drawText(text, tx + textSize * 0.2f, ty + th * 0.8f, textPaint);
        }
    }
}