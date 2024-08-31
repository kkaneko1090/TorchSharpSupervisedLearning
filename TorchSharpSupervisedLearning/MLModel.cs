using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace TorchSharpSupervisedLearning
{
    public class MLModel : Module<Tensor, Tensor>
    {
        #region メンバ変数
        /// <summary>
        /// 畳み込み層1
        /// </summary>
        private Conv2d _conv1;
        /// <summary>
        /// 畳み込み層2
        /// </summary>
        private Conv2d _conv2;
        /// <summary>
        /// 全結合層1
        /// </summary>
        private Linear _linear1;
        /// <summary>
        /// 全結合層2
        /// </summary>
        private Linear _linear2;
        /// <summary>
        /// 隠れ層のサイズ
        /// </summary>
        private int _hiddenLayerSize = 32;

        /// <summary>
        /// デバイス（CPU or GPU）
        /// </summary>
        private Device _device = CPU;
        #endregion

        /// <summary>
        /// コンストラクタ
        /// </summary>
        /// <param name="inputSize">入力する画像のサイズ</param>
        /// <param name="outputSize">出力するベクトルのサイズ</param>
        public MLModel(int[] inputSize, int outputSize) : base("CNN")
        {
            //ダミーの入力データを作成（各層の次元の初期化に使用）
            Tensor dammyInput = zeros([inputSize[0], inputSize[1]]).unsqueeze(0).unsqueeze(0);
            //畳み込み層の初期化
            _conv1 = Conv2d(in_channels: 1, out_channels: 16, kernelSize: 8, stride: 2);
            _conv2 = Conv2d(in_channels: 16, out_channels: 16, kernelSize: 8, stride: 2);
            //ダミーの畳み込み層の出力
            Tensor dammyConvOutput = _conv1.forward(dammyInput); //畳み込み層1
            dammyConvOutput = _conv2.forward(dammyConvOutput); //畳み込み層2
            dammyConvOutput = flatten(dammyConvOutput, start_dim: 1); //平滑化
            //全結合層の初期化
            _linear1 = Linear(inputSize: dammyConvOutput.shape[1], outputSize: _hiddenLayerSize);
            _linear2 = Linear(inputSize: _hiddenLayerSize, outputSize: outputSize);

            //コンポーネントの登録
            RegisterComponents();


            //GPUを使用できるか
            if (torch.cuda.is_available()) _device = CUDA; //GPUを活用
            //デバイスに転送
            this.to(_device); 
        }

        /// <summary>
        /// 順伝播処理のオーバーライド
        /// </summary>
        /// <param name="input">入力データ</param>
        /// <returns></returns>
        public override Tensor forward(Tensor input)
        {
            //畳み込み
            var x = relu(_conv1.forward(input));　//活性化関数はReLUを使用
            x = relu(_conv2.forward(x));
            //平坦化
            x = torch.flatten(x, start_dim: 1);
            //全結合層
            x = relu(_linear1.forward(x));
            x = softmax(_linear2.forward(x), dim:1); //クラス分類のためSoftmax関数
            return x;
        }

        /// <summary>
        /// バッチ学習
        /// </summary>
        /// <param name="dataset">教師データのリスト</param>
        /// <param name="epochCount">エポック数</param>
        /// <param name="batchSize">バッチサイズ</param>
        public void TrainOnBatch(List<(Tensor input, Tensor output)> dataset, int epochCount, int batchSize)
        {
            //オプティマザの初期化
            var optimizer = optim.Adam(parameters: this.parameters(), lr: 0.001); //学習率を0.001に設定
            //損失関数
            var loss = CrossEntropyLoss();

            //エポック数だけ繰り返し
            for (int epoch = 0; epoch < epochCount; epoch++)
            {
                //バッチ取り出し
                var batcheArray = Utility.GetBatch(dataset, batchSize);

                //バッチの繰り返し
                for (int batch = 0; batch < batcheArray.Length; batch++)
                {
                    //入力データ
                    var input = batcheArray[batch].input.to(_device);
                    //出力データ
                    var output = batcheArray[batch].output.to(_device);
                    
                    //オプティマイザの勾配を初期化
                    optimizer.zero_grad();
                    //推論
                    var predicted = this.forward(input);
                    //残差
                    var error = loss.forward(predicted, output);
                    //逆伝播
                    error.backward();
                    optimizer.step();

                    Console.WriteLine(error.ToSingle());
                }

                //メモリ解放
                GC.Collect();
            }
        }

        /// <summary>
        /// 推論
        /// </summary>
        /// <param name="input">単一の入力データ</param>
        /// <returns></returns>
        public (int index, float probability) Predict(Tensor input) 
        {
            //順伝播
            Tensor output =  this.forward(input.unsqueeze(0).to(_device)).squeeze(0);
            //配列に変換
            float[] array = new float[output.shape[0]];
            for(int i = 0; i < array.Length; i++) array[i] = output[i].ToSingle();
            //最大値をとるインデックスを取得
            int maxIndex = Array.IndexOf(array, array.Max());
            return (maxIndex, array[maxIndex]);
        }
    }
}
