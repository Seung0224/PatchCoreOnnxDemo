using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace PatchCoreOnnxDemo
{
    class PreprocessConfig
    {
        public int resize = 256;                   // 짧은 변 기준 리사이즈
        public int crop = 224;                     // 센터 크롭 크기
        public float[] mean = { 0.485f, 0.456f, 0.406f };
        public float[] std = { 0.229f, 0.224f, 0.225f };
    }

    class Program
    {
        static void Main(string[] args)
        {
            string baseDir = @"C:\Users\제이스텍\source\repos\PatchCoreOnnxDemo";
            string exportDir = args.Length > 0 ? args[0] : Path.Combine(baseDir, "export");
            string imagePath = args.Length > 1 ? args[1] : Path.Combine(baseDir, "sample.png");
            string onnxPath = Path.Combine(exportDir, "wrn50_l2l3.onnx");
            string cfgPath = Path.Combine(exportDir, "csharp_config.json");

            if (!File.Exists(onnxPath)) throw new FileNotFoundException("ONNX not found", onnxPath);
            if (!File.Exists(imagePath)) throw new FileNotFoundException("Image not found", imagePath);

            // 1) 전처리 설정 로드
            PreprocessConfig pp = LoadPreprocess(cfgPath);

            // 2) 이미지 전처리 → [1,3,224,224]
            DenseTensor<float> input = PreprocessToCHW(imagePath, pp);

            // 3) ONNX 실행
            SessionOptions so = CreateSessionOptions();
            using (var session = new InferenceSession(onnxPath, so))
            {
                string inputName = session.InputMetadata.Keys.First();
                NamedOnnxValue inputValue = NamedOnnxValue.CreateFromTensor(inputName, input);

                using (var results = session.Run(new[] { inputValue }))
                {
                    Console.OutputEncoding = Encoding.UTF8;
                    Console.WriteLine("== ONNX Outputs ==");
                    foreach (var r in results)
                    {
                        var t = r.AsTensor<float>();

                        // ReadOnlySpan<int> -> int[] 로 복사
                        var dims = new int[t.Dimensions.Length];
                        for (int i = 0; i < dims.Length; i++)
                            dims[i] = t.Dimensions[i];

                        Console.WriteLine("- {0}: [{1}]", r.Name, string.Join(",", dims));
                    }
                }
            }

            Console.WriteLine("\n[OK] Step1 완료 — ONNX 호출 및 출력 shape 확인");
        }

        private static SessionOptions CreateSessionOptions()
        {
            var so = new SessionOptions();
            try
            {
                // CUDA 환경일 경우: so.AppendExecutionProvider_CUDA();
                // 현재는 CPU fallback 기본
            }
            catch { }
            return so;
        }

        private static PreprocessConfig LoadPreprocess(string cfgPath)
        {
            PreprocessConfig def = new PreprocessConfig();
            try
            {
                if (!File.Exists(cfgPath)) return def;
                using (FileStream fs = File.OpenRead(cfgPath))
                {
                    using (JsonDocument doc = JsonDocument.Parse(fs))
                    {
                        JsonElement ppEl;
                        if (!doc.RootElement.TryGetProperty("preprocess", out ppEl))
                            return def;

                        PreprocessConfig pp = new PreprocessConfig();
                        JsonElement rz, cr, mn, sd;
                        if (ppEl.TryGetProperty("resize", out rz) && rz.TryGetInt32(out int rv)) pp.resize = rv;
                        if (ppEl.TryGetProperty("crop", out cr) && cr.TryGetInt32(out int cv)) pp.crop = cv;
                        if (ppEl.TryGetProperty("mean", out mn) && mn.ValueKind == JsonValueKind.Array)
                            pp.mean = mn.EnumerateArray().Select(x => (float)x.GetDouble()).ToArray();
                        if (ppEl.TryGetProperty("std", out sd) && sd.ValueKind == JsonValueKind.Array)
                            pp.std = sd.EnumerateArray().Select(x => (float)x.GetDouble()).ToArray();
                        return pp;
                    }
                }
            }
            catch
            {
                return def;
            }
        }

        private static DenseTensor<float> PreprocessToCHW(string imagePath, PreprocessConfig pp)
        {
            using (Bitmap src = new Bitmap(imagePath))
            using (Bitmap resized = ResizeKeepAspect(src, pp.resize))
            using (Bitmap cropped = CenterCrop(resized, pp.crop, pp.crop))
            {
                int H = pp.crop, W = pp.crop;
                var tensor = new DenseTensor<float>(new[] { 1, 3, H, W });

                BitmapData bmpData = cropped.LockBits(new Rectangle(0, 0, W, H), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
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
                                // Format24bppRgb = B,G,R
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

        private static Bitmap ResizeKeepAspect(Bitmap src, int shortSide)
        {
            int ow = src.Width, oh = src.Height;
            float scale = (float)shortSide / Math.Min(ow, oh);
            int nw = (int)Math.Round(ow * scale);
            int nh = (int)Math.Round(oh * scale);

            Bitmap dst = new Bitmap(nw, nh, PixelFormat.Format24bppRgb);
            using (Graphics g = Graphics.FromImage(dst))
            {
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.CompositingQuality = CompositingQuality.HighQuality;
                g.SmoothingMode = SmoothingMode.HighQuality;
                g.DrawImage(src, new Rectangle(0, 0, nw, nh), new Rectangle(0, 0, ow, oh), GraphicsUnit.Pixel);
            }
            return dst;
        }

        private static Bitmap CenterCrop(Bitmap src, int w, int h)
        {
            int x = Math.Max(0, (src.Width - w) / 2);
            int y = Math.Max(0, (src.Height - h) / 2);

            Bitmap dst = new Bitmap(w, h, PixelFormat.Format24bppRgb);
            using (Graphics g = Graphics.FromImage(dst))
            {
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.DrawImage(src, new Rectangle(0, 0, w, h), new Rectangle(x, y, w, h), GraphicsUnit.Pixel);
            }
            return dst;
        }
    }
}
