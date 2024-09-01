using Microsoft.Win32;
using SkiaSharp;
using System.Drawing;
using System.IO;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using TorchSharp;
using static TorchSharp.torch;

namespace TorchSharpSupervisedLearning
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();

            //キャンバスのペンサイズを設定
            cnvDrawingArea.DefaultDrawingAttributes.Width = 15;
            cnvDrawingArea.DefaultDrawingAttributes.Height = 15;

            //推論ボタンを機能停止
            btnPredict.IsEnabled = false;
        }

        /// <summary>
        /// 機械学習モデル
        /// </summary>
        MLModel _model;
        /// <summary>
        /// データのラベル
        /// </summary>
        string[] _labels;
        /// <summary>
        /// 画像のサイズ
        /// </summary>
        int[] _imageSize = [128, 128];

        /// <summary>
        /// Canvasのリセットボタン
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btnReset_Click(object sender, RoutedEventArgs e)
        {
            cnvDrawingArea.Strokes.Clear(); //ストロークのクリア
        }

        /// <summary>
        /// 推論ボタン
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btnPredict_Click(object sender, RoutedEventArgs e)
        {
            //CanvasをBitmapに変換
            Bitmap bitmap = Utility.InkCanvasToBitmap(cnvDrawingArea, _imageSize);
            //Tensorに変換して推論
            (int labelIndex, float probability) = _model.Predict(Utility.BitmapToTensor(bitmap));
            //フォームに推論結果と確立を表示
            txtPredicted.Text = "Predicted:  " + _labels[labelIndex] + string.Format(" ({0} %)", probability * 100);
        }

        /// <summary>
        /// 学習ボタン
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btnTrain_Click(object sender, RoutedEventArgs e)
        {
            //フォルダ選択ダイアログ
            OpenFolderDialog dialog = new OpenFolderDialog();
            dialog.ShowDialog();//表示

            //フォルダが1つだけ選択されたとき
            if(!dialog.Multiselect && dialog.FolderName != null)
            {
                //フォルダ名を取得
                string folderName = dialog.FolderName;
                //訓練用データセットとラベル
                (var dataset, _labels) = Utility.LoadDataset(folderName, _imageSize);
                //モデルの初期化
                _model = new MLModel(inputSize: _imageSize, outputSize: _labels.Length);//ラベル数から出力次元を決定
                //バッチ学習
                _model.TrainOnBatch(dataset: dataset, epochCount: 50, batchSize: 128);
                //推論ボタンを有効にする
                btnPredict.IsEnabled = true; 
            }
        }


     
    }
}