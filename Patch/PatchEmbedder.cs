using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Linq;

namespace PatchCoreOnnxDemo
{
    public sealed class PatchEmbeddings
    {
        public int GridH;   // 14
        public int GridW;   // 14
        public int Dim;     // 1536
        public int Patches; // 196
        public float[] RowsRowMajor; // length = Patches * Dim
    }

    public static class PatchEmbedder
    {
        public static PatchEmbeddings Build(Tensor<float> l2t, Tensor<float> l3t)
        {
            DenseTensor<float> l2 = l2t as DenseTensor<float> ?? ToDense(l2t);
            DenseTensor<float> l3 = l3t as DenseTensor<float> ?? ToDense(l3t);

            CheckShape4D(l2, "l2");
            CheckShape4D(l3, "l3");

            int c2 = l2.Dimensions[1], h2 = l2.Dimensions[2], w2 = l2.Dimensions[3];
            int c3 = l3.Dimensions[1], h3 = l3.Dimensions[2], w3 = l3.Dimensions[3];

            float[] l2CHW = ToCHWArray(l2);
            float[] l3CHW = ToCHWArray(l3);

            if (h2 != h3 || w2 != w3)
                l2CHW = ResizeBilinearCHW(l2CHW, c2, h2, w2, h3, w3);

            float[] catCHW = ConcatChannelsCHW(l2CHW, c2, l3CHW, c3, h3, w3);
            int dim = c2 + c3;
            int patches = h3 * w3;

            float[] rows = CHW_to_PatchesRowMajor(catCHW, dim, h3, w3);
            L2NormalizeRowsInPlace(rows, patches, dim);

            PatchEmbeddings pe = new PatchEmbeddings();
            pe.GridH = h3; pe.GridW = w3; pe.Dim = dim; pe.Patches = patches;
            pe.RowsRowMajor = rows;
            return pe;
        }

        private static DenseTensor<float> ToDense(Tensor<float> t)
        {
            int[] dims = new int[t.Dimensions.Length];
            for (int i = 0; i < dims.Length; i++) dims[i] = t.Dimensions[i];

            float[] data = t.ToArray();
            try { return new DenseTensor<float>(data, dims); }
            catch
            {
                DenseTensor<float> dense = new DenseTensor<float>(dims);
                try
                {
                    var span = dense.Buffer.Span;
                    for (int i = 0; i < data.Length; i++) span[i] = data[i];
                }
                catch
                {
                    for (int i = 0; i < data.Length; i++) dense.SetValue(i, data[i]);
                }
                return dense;
            }
        }

        private static void CheckShape4D(DenseTensor<float> t, string name)
        {
            if (t.Dimensions.Length != 4 || t.Dimensions[0] != 1)
                throw new ArgumentException(name + " must be shape [1,C,H,W]");
        }

        private static float[] ToCHWArray(DenseTensor<float> t)
        {
            int c = t.Dimensions[1], h = t.Dimensions[2], w = t.Dimensions[3];
            float[] arr = new float[c * h * w];
            int idx = 0;
            for (int ch = 0; ch < c; ch++)
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                        arr[idx++] = t[0, ch, y, x];
            return arr;
        }

        private static float[] ResizeBilinearCHW(float[] src, int c, int h, int w, int nh, int nw)
        {
            float[] dst = new float[c * nh * nw];
            for (int ch = 0; ch < c; ch++)
            {
                int srcOff = ch * h * w;
                int dstOff = ch * nh * nw;
                for (int y = 0; y < nh; y++)
                {
                    float gy = ((y + 0.5f) * h / (float)nh) - 0.5f;
                    int y0 = (int)Math.Floor(gy);
                    int y1 = y0 + 1;
                    float ly = gy - y0;
                    if (y0 < 0) { y0 = 0; y1 = 0; ly = 0f; }
                    else if (y1 >= h) { y0 = h - 1; y1 = h - 1; ly = 0f; }

                    for (int x = 0; x < nw; x++)
                    {
                        float gx = ((x + 0.5f) * w / (float)nw) - 0.5f;
                        int x0 = (int)Math.Floor(gx);
                        int x1 = x0 + 1;
                        float lx = gx - x0;
                        if (x0 < 0) { x0 = 0; x1 = 0; lx = 0f; }
                        else if (x1 >= w) { x0 = w - 1; x1 = w - 1; lx = 0f; }

                        float v00 = src[srcOff + y0 * w + x0];
                        float v01 = src[srcOff + y0 * w + x1];
                        float v10 = src[srcOff + y1 * w + x0];
                        float v11 = src[srcOff + y1 * w + x1];

                        float vx0 = v00 + (v01 - v00) * lx;
                        float vx1 = v10 + (v11 - v10) * lx;
                        float v = vx0 + (vx1 - vx0) * ly;

                        dst[dstOff + y * nw + x] = v;
                    }
                }
            }
            return dst;
        }

        private static float[] ConcatChannelsCHW(float[] a, int cA, float[] b, int cB, int h, int w)
        {
            float[] cat = new float[(cA + cB) * h * w];
            int hw = h * w;
            for (int ch = 0; ch < cA; ch++)
                Array.Copy(a, ch * hw, cat, ch * hw, hw);
            for (int ch = 0; ch < cB; ch++)
                Array.Copy(b, ch * hw, cat, (cA + ch) * hw, hw);
            return cat;
        }

        private static float[] CHW_to_PatchesRowMajor(float[] chw, int c, int h, int w)
        {
            int patches = h * w;
            float[] rows = new float[patches * c];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    int row = y * w + x;
                    int rowOff = row * c;
                    for (int ch = 0; ch < c; ch++)
                    {
                        int chwIndex = ch * h * w + y * w + x;
                        rows[rowOff + ch] = chw[chwIndex];
                    }
                }
            return rows;
        }

        private static void L2NormalizeRowsInPlace(float[] rows, int rowsCnt, int dim)
        {
            for (int r = 0; r < rowsCnt; r++)
            {
                int off = r * dim;
                double sum = 0.0;
                for (int j = 0; j < dim; j++) sum += rows[off + j] * rows[off + j];
                double norm = Math.Sqrt(sum);
                if (norm > 0)
                {
                    float inv = (float)(1.0 / norm);
                    for (int j = 0; j < dim; j++) rows[off + j] *= inv;
                }
            }
        }
    }
}
