using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace PatchCoreOnnxDemo
{
    public static class ImagePreprocessor
    {
        public static DenseTensor<float> PreprocessToCHW(string imagePath, PreprocessConfig pp)
        {
            using (Bitmap src = new Bitmap(imagePath))
            using (Bitmap resized = BitmapUtils.ResizeKeepAspect(src, pp.resize))
            using (Bitmap cropped = BitmapUtils.CenterCrop(resized, pp.crop, pp.crop))
            {
                int H = pp.crop, W = pp.crop;
                var tensor = new DenseTensor<float>(new[] { 1, 3, H, W });

                BitmapData bmpData = cropped.LockBits(
                    new Rectangle(0, 0, W, H),
                    ImageLockMode.ReadOnly,
                    PixelFormat.Format24bppRgb);

                try
                {
                    unsafe
                    {
                        byte* scan0 = (byte*)bmpData.Scan0.ToPointer();
                        int stride = bmpData.Stride;

                        for (int y = 0; y < H; y++)
                        {
                            byte* row = scan0 + y * stride;
                            for (int x = 0; x < W; x++)
                            {
                                byte b = row[x * 3 + 0];
                                byte g = row[x * 3 + 1];
                                byte r = row[x * 3 + 2];

                                float rf = (r / 255f - pp.mean[0]) / pp.std[0];
                                float gf = (g / 255f - pp.mean[1]) / pp.std[1];
                                float bf = (b / 255f - pp.mean[2]) / pp.std[2];

                                tensor[0, 0, y, x] = rf;
                                tensor[0, 1, y, x] = gf;
                                tensor[0, 2, y, x] = bf;
                            }
                        }
                    }
                }
                finally
                {
                    cropped.UnlockBits(bmpData);
                }

                return tensor;
            }
        }
    }
}
