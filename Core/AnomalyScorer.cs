using System;
using System.Linq;

namespace PatchCoreOnnxDemo
{
    public sealed class AnomalyResult
    {
        public float ImageScore;    // 패치 min-dist의 max
        public float[] PatchMin;    // 길이 = patches
        public bool IsAnomaly;
    }

    public static class AnomalyScorer
    {
        public static AnomalyResult ScoreImage(
            float[] rowsRowMajor, int patches, int d,
            float[] gallery, int ntotal,
            string metric, float threshold)
        {
            if (rowsRowMajor == null || rowsRowMajor.Length == 0)
                throw new ArgumentException("rowsRowMajor is empty");
            if (gallery == null || gallery.Length == 0)
                throw new ArgumentException("gallery is empty");
            if (d <= 0) throw new ArgumentException("d must be > 0");
            if (ntotal <= 0) throw new ArgumentException("ntotal must be > 0");

            // rowsRowMajor 길이로부터 patches 재확인 (불일치 시 재계산)
            if (rowsRowMajor.Length % d != 0)
                throw new InvalidOperationException(
                    $"rowsRowMajor length {rowsRowMajor.Length} not divisible by dim {d}");
            int inferredPatches = rowsRowMajor.Length / d;
            if (patches != inferredPatches)
            {
                Console.WriteLine($"[WARN] patches({patches}) → inferred({inferredPatches})로 수정");
                patches = inferredPatches;
            }

            // gallery 길이로부터 dim 재확인
            if (gallery.Length % ntotal != 0)
                throw new InvalidOperationException(
                    $"gallery length {gallery.Length} not divisible by ntotal {ntotal}");
            int inferredDim = gallery.Length / ntotal;
            if (inferredDim != d)
                throw new InvalidOperationException(
                    $"[DIM MISMATCH] patch dim(d)={d}, gallery dim={inferredDim} — " +
                    $"wrn/임베딩과 gallery 차원이 다릅니다. " +
                    $"(export 산출물과 현재 ONNX/임베딩 경로가 일치하는지 확인)");

            var patchMin = new float[patches];
            Distance.RowwiseMinDistances(rowsRowMajor, patches, d, gallery, ntotal, metric, patchMin);

            float imgScore = patchMin.Max();
            bool isAnom = imgScore > threshold;

            return new AnomalyResult
            {
                ImageScore = imgScore,
                PatchMin = patchMin,
                IsAnomaly = isAnom
            };
        }
    }
}
