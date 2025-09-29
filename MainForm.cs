using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Drawing;
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

        // 최근 추론 결과(히트맵용)
        private float[] _lastPatchMin;     // length = GridH*GridW (ex. 196)
        private string _lastImagePath;

        // ==== UI ====
        private Button btnLoadExport;
        private Button btnRunImage;
        private Button btnOverlay;
        private Label lblExport;
        private Label lblResult;

        public MainForm()
        {
            InitializeComponent();

            Text = "PatchCore L3-only (.NET 4.8.1)";
            Width = 720;
            Height = 260;

            btnLoadExport = new Button
            {
                Text = "1) Export 폴더 선택...",
                Left = 20,
                Top = 20,
                Width = 220,
                Height = 32
            };
            btnLoadExport.Click += BtnLoadExport_Click;
            Controls.Add(btnLoadExport);

            btnRunImage = new Button
            {
                Text = "2) 이미지 선택 & 추론",
                Left = 20,
                Top = 70,
                Width = 220,
                Height = 32,
                Enabled = false
            };
            btnRunImage.Click += BtnRunImage_Click;
            Controls.Add(btnRunImage);

            btnOverlay = new Button
            {
                Text = "3) 원본+Heatmap 저장",
                Left = 20,
                Top = 120,
                Width = 220,
                Height = 32,
                Enabled = false
            };
            btnOverlay.Click += BtnOverlay_Click;
            Controls.Add(btnOverlay);

            lblExport = new Label
            {
                AutoSize = false,
                Left = 260,
                Top = 25,
                Width = 420,
                Height = 24,
                Text = "Export: (미선택)"
            };
            Controls.Add(lblExport);

            lblResult = new Label
            {
                AutoSize = false,
                Left = 260,
                Top = 70,
                Width = 420,
                Height = 100,
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
                        _exportDir = fbd.SelectedPath;

                        // 아티팩트 로드
                        _artifacts = Artifacts.Load(_exportDir);

                        _session?.Dispose();
                        _session = OnnxRunner.CreateSession(_artifacts.OnnxPath);

                        // 상태 초기화
                        _lastPatchMin = null;
                        _lastImagePath = null;

                        lblExport.Text = $"Export: {_exportDir}";
                        lblResult.Text = "결과: (준비됨)";
                        btnRunImage.Enabled = true;
                        btnOverlay.Enabled = false;
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show(this, ex.Message, "Export 로드 실패", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        btnRunImage.Enabled = false;
                        btnOverlay.Enabled = false;
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
                btnOverlay.Enabled = false;
                lblResult.Text = "결과: 추론 중...";

                try
                {
                    var imagePath = ofd.FileName;

                    // 백그라운드에서 추론 실행
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
                            "ip", patchMin);

                        float imgScore = patchMin.Max();
                        bool isAnomaly = imgScore > _artifacts.Threshold;

                        return new InferOut
                        {
                            ImageScore = imgScore,
                            IsAnomaly = isAnomaly,
                            ImagePath = imagePath,
                            PatchMin = patchMin
                        };
                    });

                    // 결과 표시 + 나중에 히트맵용으로 저장
                    _lastPatchMin = result.PatchMin;
                    _lastImagePath = result.ImagePath;

                    var label = result.IsAnomaly ? "NotGood" : "Good";
                    lblResult.Text = $"결과: score={result.ImageScore:F6} / thr={_artifacts.Threshold:F6} → {label}\n({Path.GetFileName(result.ImagePath)})";

                    btnOverlay.Enabled = true;
                }
                catch (Exception ex)
                {
                    MessageBox.Show(this, ex.Message, "추론 실패", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    lblResult.Text = "결과: 오류";
                    btnOverlay.Enabled = false;
                }
                finally
                {
                    btnRunImage.Enabled = true;
                }
            }
        }

        private void BtnOverlay_Click(object sender, EventArgs e)
        {
            try
            {
                if (_lastPatchMin == null || _lastImagePath == null)
                {
                    MessageBox.Show(this, "먼저 추론을 실행해 주세요.", "알림",
                        MessageBoxButtons.OK, MessageBoxIcon.Information);
                    return;
                }

                using (var bmp = new Bitmap(_lastImagePath))
                using (var ov = HeatmapOverlay.MakeOverlay(
                    bmp, _lastPatchMin, _artifacts.GridH, _artifacts.GridW,
                    clipQ: 0.98f, gamma: 1.8f, alphaMin: 0.02f, alphaMax: 0.50f))
                {
                    string save = Path.ChangeExtension(_lastImagePath, ".overlay.png");
                    ov.Save(save);
                    MessageBox.Show(this, $"저장됨: {save}", "완료",
                        MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(this, ex.Message, "히트맵 생성 실패",
                    MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private sealed class InferOut
        {
            public float ImageScore { get; set; }
            public bool IsAnomaly { get; set; }
            public string ImagePath { get; set; }
            public float[] PatchMin { get; set; }
        }
    }
}
