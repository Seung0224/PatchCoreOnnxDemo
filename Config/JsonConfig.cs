using System.IO;
using System.Linq;
using System.Text.Json;

namespace PatchCoreOnnxDemo
{
    public static class JsonConfig
    {
        public static PreprocessConfig LoadPreprocess(string cfgPath)
        {
            PreprocessConfig def = new PreprocessConfig();
            try
            {
                if (!File.Exists(cfgPath)) return def;
                using (FileStream fs = File.OpenRead(cfgPath))
                using (JsonDocument doc = JsonDocument.Parse(fs))
                {
                    JsonElement ppEl;
                    if (!doc.RootElement.TryGetProperty("preprocess", out ppEl))
                        return def;

                    PreprocessConfig pp = new PreprocessConfig();
                    JsonElement rz, cr, mn, sd;
                    int rv, cv;
                    if (ppEl.TryGetProperty("resize", out rz) && rz.TryGetInt32(out rv)) pp.resize = rv;
                    if (ppEl.TryGetProperty("crop", out cr) && cr.TryGetInt32(out cv)) pp.crop = cv;
                    if (ppEl.TryGetProperty("mean", out mn) && mn.ValueKind == JsonValueKind.Array)
                        pp.mean = mn.EnumerateArray().Select(x => (float)x.GetDouble()).ToArray();
                    if (ppEl.TryGetProperty("std", out sd) && sd.ValueKind == JsonValueKind.Array)
                        pp.std = sd.EnumerateArray().Select(x => (float)x.GetDouble()).ToArray();
                    return pp;
                }
            }
            catch
            {
                return def;
            }
        }
    }
}
