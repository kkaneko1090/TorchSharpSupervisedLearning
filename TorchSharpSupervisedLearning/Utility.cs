using System.Drawing;
using System.IO;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using TorchSharp;
using static TorchSharp.torch;

namespace TorchSharpSupervisedLearning
{
    public static class Utility
    {
        /// <summary>
        /// 学習データの読み込み
        /// </summary>
        /// <param name="folderPath">フォルダパス</param>
        /// <returns></returns>
        public static (List<(Tensor input, Tensor output)> dataset, string[] labels) LoadDataset(string folderPath, int[] imageSize)
        {
            //サブフォルダパスを取得
            var subFolders = System.IO.Directory.GetDirectories(folderPath, "*", System.IO.SearchOption.TopDirectoryOnly);
            //サブフォルダ名をラベルとして認識
            string[] labels = subFolders.Select(x => Path.GetFileName(x)).ToArray();
            //戻り値となる学習データセット
            var dataset = new List<(Tensor input, Tensor output)>();

            //ラベルの繰り返し
            for (int labelIndex = 0; labelIndex < labels.Length; labelIndex++)
            {
                //このラベル（フォルダ）内の画像のパスを取得
                string[] files = Directory.GetFiles(subFolders[labelIndex]);

                //パスの繰り返し
                foreach (var path in files)
                {
                    //画像を読み込み，リサイズする
                    Bitmap bmp = ResizeBitmap(new Bitmap(path), imageSize[0], imageSize[1]);
                    //画像をTensorに変換
                    Tensor imageTensor = BitmapToTensor(bmp);

                    //ラベルを配列化
                    float[] labelArray = new float[labels.Length];
                    labelArray[labelIndex] = 1;
                    //Tensorに変換
                    Tensor labelTensor = tensor(rawArray: labelArray, dimensions: new long[] { labelArray.Length }, dtype: float32);

                    //リストに追加
                    dataset.Add((imageTensor, labelTensor));
                }
            }


            return (dataset, labels);
        }

        /// <summary>
        /// Bitmapのリサイズ
        /// </summary>
        /// <param name="originalBitmap"></param>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <returns></returns>
        static Bitmap ResizeBitmap(Bitmap originalBitmap, int width, int height)
        {
            // 指定されたサイズで新しいBitmapを作成
            Bitmap resizedBitmap = new Bitmap(width, height);
            // Graphicsオブジェクトを使ってリサイズ
            using (Graphics graphics = Graphics.FromImage(resizedBitmap))
            {
                //背景の初期化
                graphics.Clear(System.Drawing.Color.White);
                //補間モードを指定
                graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.High;
                //新しいサイズで描画
                graphics.DrawImage(originalBitmap, 0, 0, width, height);
                return resizedBitmap;
            }

        }

        /// <summary>
        /// BitmapをTensorに変換
        /// </summary>
        /// <param name="bitmap">Bitmap</param>
        /// <returns></returns>
        public static Tensor BitmapToTensor(Bitmap bitmap)
        {
            //データの次元
            int width = bitmap.Width;
            int height = bitmap.Height;
            int channels = 1; // 白黒のため
            // ピクセルデータを格納する配列を作成
            float[] imageData = new float[width * height * channels];

            // Bitmapからピクセルデータを取得
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    System.Drawing.Color pixel = bitmap.GetPixel(x, y);
                    int index = (y * width + x) * channels;
                    imageData[index] = (pixel.R + pixel.G + pixel.B) / 3 / 255.0f;
                }
            }

            // Tensorを作成 (チャンネル、高さ、幅の順序で配置)
            var tensor = torch.tensor(rawArray: imageData, dimensions: new long[] { channels, height, width }, dtype: float32);
            return tensor;
        }

        /// <summary>
        /// 訓練データをバッチごとに分割して返す
        /// </summary>
        /// <param name="dataset">訓練データセット</param>
        /// <param name="batchSize">バッチサイズ</param>
        /// <returns></returns>
        public static (Tensor input, Tensor output)[] GetBatch(List<(Tensor input, Tensor output)> dataset, int batchSize)
        {
            //バッチに格納するデータの順番
            int[] order = Enumerable.Range(0, dataset.Count).ToArray();
            //バッチ数
            int batchCount = (int)Math.Ceiling((decimal)dataset.Count / batchSize);
            //バッチの集合（リスト）
            var batchList = new List<(Tensor input, Tensor output)>();

            //バッチの繰り返し
            for (int batchIndex = 0; batchIndex < batchCount; batchIndex++)
            {
                //バッチ
                List<(Tensor, Tensor)> batch = new List<(Tensor, Tensor)>();

                //データの繰り返し
                for (int i = 0; i < batchSize; i++)
                {
                    //データのインデックス
                    int dataIndex = batchIndex * batchSize + i;
                    //インデックスが範囲内のとき，バッチに追加
                    if (dataIndex < dataset.Count) batch.Add(dataset[order[dataIndex]]);
                }

                //バッチ内のTensorを1つに結合
                Tensor tempInput = torch.stack(batch.Select(x => x.Item1));
                Tensor tempOutput = torch.stack(batch.Select(x => x.Item2));
                //バッチリストに格納
                batchList.Add((tempInput, tempOutput));
            }

            //配列に変換して返す
            return batchList.ToArray();
        }

        /// <summary>
        /// InkCanvasをBitmapに変換
        /// </summary>
        /// <param name="canvas">InkCanvas</param>
        /// <param name="bitmapSize">Bitmapのサイズ</param>
        /// <returns></returns>
        public static Bitmap InkCanvasToBitmap(InkCanvas canvas, int[] bitmapSize)
        {
            // InkCanvasのサイズを取得
            int width = (int)canvas.ActualWidth;
            int height = (int)canvas.ActualHeight;
            // RenderTargetBitmapに変換
            RenderTargetBitmap renderBitmap = new RenderTargetBitmap(width, height, 96, 96, PixelFormats.Pbgra32);
            renderBitmap.Render(canvas);

            //Bitmapに変換
            using (var stream = new MemoryStream())
            {
                PngBitmapEncoder encoder = new PngBitmapEncoder();
                encoder.Frames.Add(BitmapFrame.Create(renderBitmap));
                encoder.Save(stream);
                Bitmap bmp = new Bitmap(stream);
                //リサイズして返す
                return ResizeBitmap(bmp, bitmapSize[0], bitmapSize[1]);
            }
        }

    }
}
