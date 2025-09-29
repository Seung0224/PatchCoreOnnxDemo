using System;

namespace PatchCoreOnnxDemo
{
    public static class Distance
    {
        // rowsA: (na x d), rowsB: (nb x d) — 모두 row-major
        // metric = "ip" (cosine/IP: A,B는 L2 정규화 전제 → 1 - dot), or "l2"
        public static void RowwiseMinDistances(
            float[] rowsA, int na, int d,
            float[] rowsB, int nb,
            string metric,
            float[] outMin) // length = na
        {
            if (metric == null) metric = "ip";
            bool useCos = metric.Equals("ip", StringComparison.OrdinalIgnoreCase);

            for (int i = 0; i < na; i++)
            {
                int offA = i * d;
                float best = float.PositiveInfinity;

                for (int j = 0; j < nb; j++)
                {
                    int offB = j * d;

                    float acc = 0f;
                    if (useCos)
                    {
                        // cosine/IP: 거리 = 1 - dot(A,B)  (A,B는 이미 L2-normalized)
                        for (int k = 0; k < d; k++) acc += rowsA[offA + k] * rowsB[offB + k];
                        float dist = 1f - acc;
                        if (dist < best) best = dist;
                    }
                    else
                    {
                        // L2: ||A - B||2
                        for (int k = 0; k < d; k++)
                        {
                            float diff = rowsA[offA + k] - rowsB[offB + k];
                            acc += diff * diff;
                        }
                        float dist = (float)Math.Sqrt(acc);
                        if (dist < best) best = dist;
                    }
                }
                outMin[i] = best;
            }
        }
    }
}
