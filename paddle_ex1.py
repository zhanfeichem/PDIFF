#coding:utf-8
import paddle.v2 as paddle
import numpy as np
#按照y=3x+1生成训练数据
def train_reader():
    data = np.array([[1, 4], [2, 7], [3, 10], [4, 13], [5, 16], [6, 19], [7, 22]])
    mydat=np.array([1,2,3,4,5,6,7])
    def reader():
        #for d in data:
            #yield d[:-1], d[-1:]
        for i in range(mydat.shape[0]):
            x=mydat[i]
	    yield [x],[3*x]
    return reader


#按照y=3x+1生成测试数据
def test_reader():
    data = ([[0.5, 2.5], [1.5, 5.5], [-2, -5], [0, 1]])
    def reader():
        for d in data:
            yield d[:-1], d[-1:]
    return reader
 
# 从数据数据集reader中提取测试数据和Label
test_data = []
test_label = []
test_data_creator = test_reader()
for item in test_data_creator():
    test_data.append((item[0], ))
    test_label.append(item[1])

#初始化Paddle
paddle.init(use_gpu=False)
 
#配置训练网络
x = paddle.layer.data(name='x', type=paddle.data_type.dense_vector(1))
y_predict = paddle.layer.fc(input=x, size=1, act=paddle.activation.Linear())
y = paddle.layer.data(name='y', type= paddle.data_type.dense_vector(1))
cost = paddle.layer.square_error_cost(input=y_predict, label=y)



#创建参数
parameters = paddle.parameters.create(cost)

#创建Trainer
optimizer = paddle.optimizer.Momentum(momentum=0)
 
trainer = paddle.trainer.SGD(cost= cost,
                            parameters=parameters,
                            update_equation=optimizer)

def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id % 1 == 0:
            print "Pass %d, Batch %d, Cost %f" % (event.pass_id, event.batch_id, event.cost)

# 开始训练
trainer.train(paddle.batch(paddle.reader.shuffle(train_reader(),buf_size=3),batch_size=2),
num_passes=10,event_handler=event_handler)
#test

probs = paddle.infer(output_layer=y_predict, parameters=parameters, input=test_data)
for i in xrange(len(probs)):
     print(test_data[i][0],"predict",probs[i][0])
				
