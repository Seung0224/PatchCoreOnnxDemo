using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace PatchCoreOnnxDemo
{
    public partial class MainForm : Form
    {
        // ==== 상태 ====
        private string _exportDir = null;
        private Artifacts _artifacts = null;
        private InferenceSession _session = null;
        private readonly PreprocessConfig _pp = new PreprocessConfig(); // 256→CenterCrop224 + ImageNet mean/std

        // ==== UI ====
        private Button btnLoadExport;
        private Button btnRunImage;
        private Label lblExport;
        private Label lblResult;

        public MainForm()
        {
            InitializeComponent();

            Text = "PatchCore L3-only (.NET 4.8.1)";
            Width = 680;
            Height = 220;

            btnLoadExport = new Button
            {
                Text = "1) Export 폴더 선택...",
                Left = 20,
                Top = 20,
                Width = 200,
                Height = 32
            };
            btnLoadExport.Click += BtnLoadExport_Click;
            Controls.Add(btnLoadExport);

            btnRunImage = new Button
            {
                Text = "2) 이미지 선택 & 추론",
                Left = 20,
                Top = 70,
                Width = 200,
                Height = 32,
                Enabled = false
            };
            btnRunImage.Click += BtnRunImage_Click;
            Controls.Add(btnRunImage);

            lblExport = new Label
            {
                AutoSize = false,
                Left = 240,
                Top = 25,
                Width = 400,
                Height = 24,
                Text = "Export: (미선택)"
            };
            Controls.Add(lblExport);

            lblResult = new Label
            {
                AutoSize = false,
                Left = 240,
                Top = 75,
                Width = 400,
                Height = 60,
                Text = "결과: -"
            };
            Controls.Add(lblResult);

            FormClosing += (s, e) => { _session?.Dispose(); };
        }

        private void BtnLoadExport_Click(object sender, EventArgs e)
        {
            using (var fbd = new FolderBrowserDialog())
            {
                fbd.Description = "wrn50_l3.onnx / gallery_f32.bin / threshold.json / meta.json 이 있는 폴더를 선택하세요.";
                if (fbd.ShowDialog(this) == DialogResult.OK)
                {
                    try
                    {
                        // 아티팩트 로드
                        _exportDir = fbd.SelectedPath;
                        _artifacts = Artifacts.Load(_exportDir); // wrn50_l3.onnx 경로/갤러리/임계값 로드
                        _session?.Dispose();
                        _session = OnnxRunner.CreateSession(_artifacts.OnnxPath);

                        lblExport.Text = $"Export: {_exportDir}";
                        btnRunImage.Enabled = true;
                        lblResult.Text = "결과: (준비됨)";
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show(this, ex.Message, "Export 로드 실패", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    }
                }
            }
        }

        private async void BtnRunImage_Click(object sender, EventArgs e)
        {
            using (var ofd = new OpenFileDialog())
            {
                ofd.Title = "추론할 이미지를 선택하세요";
                ofd.Filter = "이미지 파일|*.png;*.jpg;*.jpeg;*.bmp";
                if (ofd.ShowDialog(this) != DialogResult.OK) return;

                btnRunImage.Enabled = false;
                lblResult.Text = "결과: 추론 중...";

                try
                {
                    var imagePath = ofd.FileName;

                    // 비동기 실행(간단히 Task.Run) — UI Freeze 방지
                    var result = await Task.Run(() =>
                    {
                        // 1) 전처리 → 입력 텐서 [1,3,224,224]
                        DenseTensor<float> input = ImagePreprocessor.PreprocessToCHW(imagePath, _pp);

                        // 2) ONNX 실행 → "layer3" 출력
                        var results = OnnxRunner.Run(_session, input);
                        var tL3 = results.First(r => r.Name == "layer3").AsTensor<float>();

                        // 3) L3-only 패치 임베딩 (196×1024) + 행별 L2 정규화
                        var pe = PatchEmbedder.BuildFromL3(tL3);

                        if (pe.Dim != _artifacts.Dim)
                            throw new InvalidDataException($"patch-dim({pe.Dim}) != gallery-dim({_artifacts.Dim})");

                        // 4) 코사인 거리(1−dot), k=1 → patch별 최소거리 → 이미지 스코어=max
                        var patchMin = new float[pe.Patches];
                        Distance.RowwiseMinDistances(
                            pe.RowsRowMajor, pe.Patches, pe.Dim,
                            _artifacts.Gallery, _artifacts.GalleryRows,
                            "ip", patchMin); // "ip" = inner product -> cosine(정규화 전제)

                        float imgScore = patchMin.Max();
                        bool isAnomaly = imgScore > _artifacts.Threshold;
                        return (imgScore, isAnomaly, imagePath);
                    });

                    var label = result.isAnomaly ? "NotGood" : "Good";
                    lblResult.Text = $"결과: score={result.imgScore:F6} / thr={_artifacts.Threshold:F6} → {label}\n({Path.GetFileName(result.imagePath)})";
                }
                catch (Exception ex)
                {
                    MessageBox.Show(this, ex.Message, "추론 실패", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    lblResult.Text = "결과: 오류";
                }
                finally
                {
                    btnRunImage.Enabled = true;
                }
            }
        }
    }
}
