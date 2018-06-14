import cirrus

def progress_callback(loss, cost, task):
  print("Current training loss:", loss, \
        "current cost ($): ", cost)

data_bucket = 'cirrus-criteo-kaggle-19b-random'
model = 'model_v1'

lr_task = cirrus.LogisticRegression(
             # number of workers and number of PSs
             n_workers = 3, n_ps = 2,
             # path to s3 bucket with input dataset
             dataset = data_bucket,
             # sgd update LR and epsilon
             learning_rate=0.01, epsilon=0.0001,
             # 
             progress_callback = progress_callback,
             # stop workload after these many seconds
             timeout = 0,
             # stop workload once we reach this loss
             threshold_loss=0,
             # resume execution from model stored in this s3 bucket
             resume_model = model,
             # aws key name
             key_name='mykey',
             # path to aws key
             key_path='/home/joao/Downloads/mykey.pem',
             # ip where ps lives
             ps_ip='ec2-34-214-232-215.us-west-2.compute.amazonaws.com',
             # username of VM
             ps_username='ubuntu',
             # choose between adagrad, sgd, nesterov, momentum
             opt_method = 'adagrad',
             # checkpoint model every x secs
             checkpoint_model = 60,
             # 
             minibatch_size=20,
             # model size
             model_bits=19,
             # whether to filter gradient weights
             use_grad_threshold=True,
             # threshold value
             grad_threshold=0.001,
             # range of training minibatches
             train_set=(0,824),
             # range of testing minibatches
             test_set=(825,840)
             )

lr_task.run()

#model, loss = lr_task.wait()
