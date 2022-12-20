MODELNAME = 'ConvLSTMEncoderDecoder'
KEYWORD = f'Density_{DATANAME}_{MODELNAME}' + '_' + datetime.now().strftime("%y%m%d%H%M")
PATH = '../save/' + KEYWORD
scaler = StandardScaler()
np.random.seed(100)
random.seed(100)
torch.manual_seed(seed=100)
data = np.load(DATAPATH)
np.random.shuffle(data)
data = data.reshape(data.shape[0], -1) #行为data.shape[0]行，列自动算出。data.shape[0]:data第一维的长度。
data = scaler.fit_transform(data)
data = data.reshape(-1, CHANNEL, HEIGHT, WIDTH)

XS, YS = getXSYS(data)  #<class 'numpy.ndarray'>
B, T, C, H, W = YS.shape
YS = YS.reshape(B * T, C * H * W )
# YS, YS_pred = scaler.inverse_transform(YS), scaler.inverse_transform(YS_pred)
YS = YS.reshape(B, T, C, H, W) #<class 'numpy.ndarray'>


"""
xx = torch.tensor(XS, dtype=torch.float32)
yy = torch.tensor(YS, dtype=torch.float32)
print(xx.shape, yy.shape)
"""
model = ConvLSTM()


def run():
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)
    shutil.copy2('Param.py', PATH)

    train_data = data[:TRAIN_DAYS * DAY_TIMESTAMP, :, :, :]
    test_data = data[TRAIN_DAYS * DAY_TIMESTAMP:, :, :, :]
    #print(data.shape, train_data.shape, test_data.shape)

    trainXS, trainYS = getXSYS(train_data)
    print('TRAIN XS YS', trainXS.shape, trainYS.shape) #<class 'numpy.ndarray'>
    train_x = torch.tensor(trainXS, dtype=torch.float32)  # torch.from_numpy(): numpy中的ndarray转化成pytorch中的tensor(张量)
    train_y = torch.tensor(trainYS, dtype=torch.float32)

    testXS, testYS = getXSYS(test_data)
    print('TEST XS YS', testXS.shape, testYS.shape) #<class 'numpy.ndarray'>

    test_x = torch.tensor(testXS, dtype=torch.float32)
    test_y = torch.tensor(testYS, dtype=torch.float32)

    #model.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    optim = torch.optim.Adam(model.parameters(), lr= 0.0001)
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_accuracy = 0.

    for i in range(EPOCH):

        optim.zero_grad()
        output, last_states = model(train_x)

        output = output.permute(1, 0, 2, 3, 4)
        loss = criterion(output, train_y)
        total_loss += loss.item()
        loss.backward() # 每次训练前将梯度重置为0
        optim.step()
        train_loss = total_loss / len(train_x)

        outputs, last_states = model(test_x)
        outputs = outputs.permute(1, 0, 2, 3, 4)
        #pred = outputs.argmax(dim=1)
        #total_accuracy += torch.eq(outputs, test_y).sum().item()
        #acc = total_accuracy / len(test_y)

        outputs = outputs.detach().numpy()
        testYS, outputs = scaler.inverse_transform(testYS.reshape(-1, 6400)), scaler.inverse_transform(outputs.reshape(-1, 6400))
        #MAE = metrics.mean_absolute_error(testYS, outputs)
        #MSE = metrics.mean_squared_error(testYS, outputs)
        #RMSE = np.sqrt(mean_squared_error(testYS, outputs))
        #MAPE = mean_absolute_percentage_error(testYS, outputs)
        MSE, RMSE, MAE, MAPE = Metrics.evaluate(testYS, outputs)
        with open(os.path.join(PATH, f'{name}_prediction_scores.txt'), 'a') as f:
            f.write(
                " MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (
                 MSE, RMSE, MAE, MAPE))
        print(" MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f" % (MSE, RMSE, MAE, MAPE))

        #precision = metrics.precision_score(testYS, outputs)
        #print(f"params : {num_params / 1e6} M")
        #summary(model, (1, 80, 80))  # summary(your_model, input_size=(channels, H, W))
        #smape = 100 / len(testYS) * np.sum(
#            2 * np.abs(outputs - testYS) / (np.abs(testYS) + np.abs(outputs)))
        #print("smape",smape)
        print(
            f'\r epoch:{i + 1}/{EPOCH} train: loss:{total_loss:.3f} test: MAE:{MAE:.3f}  MSE:{MSE:.3f}  ' \
            f'RMSE:{RMSE:.3f}', end='')

        print('Model Training Ended ', time.ctime())
        np.save(PATH + '/' + MODELNAME + '_groundtruth.npy', testYS)
        torch.save(model.state_dict(), 'net_params.pkl')#保存训练文件net_params.pkl
        #state_dict 是一个简单的python的字典对象,将每一层与它的对应参数建立映射关系

        

if __name__ == '__main__':
    run()
