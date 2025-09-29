using System;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace PatchCoreOnnxDemo
{
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

            // 1) 전처리 설정
            PreprocessConfig pp = JsonConfig.LoadPreprocess(cfgPath);

            // 2) 이미지 → 텐서 [1,3,224,224]
            DenseTensor<float> input = ImagePreprocessor.PreprocessToCHW(imagePath, pp);

            // 3) ONNX 실행
            using (var session = OnnxRunner.CreateSession(onnxPath))
            using (var results = OnnxRunner.Run(session, input))
            {
                Console.OutputEncoding = Encoding.UTF8;
                Console.WriteLine("== ONNX Outputs ==");

                // 출력 shape 로그
                foreach (var r in results)
                {
                    var t = r.AsTensor<float>();
                    var dims = t.Dimensions.ToArray(); // ReadOnlySpan<int> → int[]
                    Console.WriteLine("- {0}: [{1}]", r.Name, string.Join(",", dims));
                }

                Console.WriteLine("\n[OK] Step1 완료 — ONNX 호출 및 출력 shape 확인");

                var tL2 = results.First(r => r.Name == "l2").AsTensor<float>();
                var tL3 = results.First(r => r.Name == "l3").AsTensor<float>();

                var pe = PatchEmbedder.Build(tL2, tL3);
                Console.WriteLine("[Step2] patches={0} (grid {1}x{2}), dim={3}",
                    pe.Patches, pe.GridH, pe.GridW, pe.Dim);
            }
        }
    }
}