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
            // === 경로 설정 ===
            string baseDir = @"C:\Users\제이스텍\source\repos\PatchCoreOnnxDemo";
            // export 폴더: wrn50_l2l3.onnx, csharp_config.json, params_meta.json, gallery_f32.bin, ids_i32.bin, (옵션) gallery_meta.json
            string exportDir = args.Length > 0 ? args[0] : Path.Combine(baseDir, "export");
            string imagePath = args.Length > 1 ? args[1] : Path.Combine(baseDir, "sample.png");

            if (!File.Exists(imagePath)) throw new FileNotFoundException("Image not found", imagePath);
            Console.OutputEncoding = Encoding.UTF8;

            // === 0) 아티팩트 로드 (ONNX 세션 + 전처리/메타 + 갤러리/IDs) ===
            var artifacts = Artifacts.Load(exportDir); // ParamsMeta/Preprocess/Gallery/Ids/Session 로드

            // === 1) 이미지 → 입력 텐서 [1,3,224,224] ===
            var pp = artifacts.Pre; // csharp_config.json에서 읽은 전처리 파라미터
            DenseTensor<float> input = ImagePreprocessor.PreprocessToCHW(imagePath, pp);

            // === 2) ONNX 실행 (l2, l3 출력 확인) ===
            using (InferenceSession session = artifacts.Session)
            using (var results = OnnxRunner.Run(session, input))
            {
                Console.WriteLine("== ONNX Outputs ==");
                foreach (var r in results)
                {
                    var t = r.AsTensor<float>();
                    var dims = t.Dimensions.ToArray();
                    Console.WriteLine("- {0}: [{1}]", r.Name, string.Join(",", dims));
                }
                Console.WriteLine("\n[OK] Step1 완료 — ONNX 호출 및 출력 shape 확인");

                var tL2 = results.First(r => r.Name == "l2").AsTensor<float>();
                var tL3 = results.First(r => r.Name == "l3").AsTensor<float>();

                // === 3) 패치 임베딩
                var pe = PatchEmbedder.Build(tL2, tL3);
                Console.WriteLine("[Step2] patches={0} (grid {1}x{2}), patch-dim={3}",
                    pe.Patches, pe.GridH, pe.GridW, pe.Dim);

                // 갤러리 정보
                Console.WriteLine("[Gallery] ntotal={0}, dim={1}", artifacts.NTotal, artifacts.Dim);

                // === [핵심] 차원 맞추기: 1536 → 1024 (l3만 사용)
                int d = pe.Dim;
                float[] rows = pe.RowsRowMajor;
                if (pe.Dim != artifacts.Dim)
                {
                    if (pe.Dim == 1536 && artifacts.Dim == 1024)
                    {
                        Console.WriteLine("[INFO] Using L3-only: slicing 1536→1024 to match gallery.");
                        rows = SliceL3(pe.RowsRowMajor, pe.Patches, pe.Dim, 1024);  // 뒤 1024 (l3)
                        d = 1024;
                    }
                    else
                    {
                        Console.WriteLine(
                            $"[ERROR] patch-dim({pe.Dim}) != gallery-dim({artifacts.Dim}) — 자동 보정 불가 조합.\n" +
                            $"갤러리를 재생성(임베딩 {pe.Dim})하거나 ONNX/임베딩을 {artifacts.Dim}로 맞추세요.");
                        return;
                    }
                }

                // === 4) 스코어/판정
                var meta = artifacts.Meta;
                var res = AnomalyScorer.ScoreImage(
                    rows, pe.Patches, d,                  // <- d는 1024로 바뀔 수 있음
                    artifacts.Gallery, artifacts.NTotal,
                    meta.Metric, meta.Threshold);

                string label = res.IsAnomaly ? "NotGood" : "Good";
                Console.WriteLine("[Step3] metric={0}, threshold={1:F6}", meta.Metric, meta.Threshold);
                Console.WriteLine("[Result] image_score={0:F6} → {1}", res.ImageScore, label);
            }
        }

        static float[] SliceL3(float[] rows1536, int patches, int dim1536, int dimKeep = 1024)
        {
            if (dim1536 < dimKeep) throw new ArgumentException("dimKeep > dim1536");
            int drop = dim1536 - dimKeep;          // 1536 - 1024 = 512
            float[] outRows = new float[patches * dimKeep];
            for (int i = 0; i < patches; i++)
            {
                int srcOff = i * dim1536 + drop;   // 뒤 1024 시작 위치
                int dstOff = i * dimKeep;
                Buffer.BlockCopy(rows1536, srcOff * sizeof(float), outRows, dstOff * sizeof(float), dimKeep * sizeof(float));
            }
            return outRows;
        }
    }
}
