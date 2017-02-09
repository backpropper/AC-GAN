require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'paths'
require 'pl'

----------------------------------------------------------------------
-- parse command-line options
opt = lapp[[
  --lr               (default 0.1)         learning rate, for SGD only
  --batchSize        (default 100)         batch size
  --beta1            (default 0.5)         momentum
  --gpu              (default 0)           on gpu 
  --noiseDim         (default 100)         dimensionality of noise vector
  --name             (default "model")     directory to save to
  --ndf              (default 64)          number of hidden units in D
  --ngf              (default 64)          number of hidden units in G
  --noise            (default "uniform")   type of noise
  --nEpochs          (default 15)          number of epochs
]]

require 'mnist'

opt.manualSeed = torch.random(1, 10000)
print('Random Seed: ' .. opt.manualSeed)
-- torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

if opt.gpu > 0 then
  require 'cutorch' 
  require 'cunn'
  cutorch.setDevice(opt.gpu)
  print('using gpu ' .. opt.gpu)
  cutorch.manualSeed(opt.manualSeed)
else
  torch.manualSeed(opt.manualSeed)
end

opt.geometry = {1, 32, 32}
opt.condDim = 10

adversarial = require 'train_test'

----------------------------------------------------------------------
-- define Discriminator
model_D = nn.Sequential()
model_D:add(nn.SpatialConvolution(1, 64, 4, 4, 2, 2, 1, 1))
model_D:add(nn.LeakyReLU(0.2, true))
model_D:add(nn.SpatialConvolution(64, 128, 4, 4, 2, 2, 1, 1))
model_D:add(nn.SpatialBatchNormalization(128))
model_D:add(nn.LeakyReLU(0.2, true))
model_D:add(nn.SpatialConvolution(128, 256, 4, 4, 2, 2, 1, 1))
model_D:add(nn.SpatialBatchNormalization(256))
model_D:add(nn.LeakyReLU(0.2, true))
model_D:add(nn.SpatialConvolution(256, 512, 4, 4, 2, 2, 1, 1))
model_D:add(nn.Reshape(512 * 2 * 2))
model_D:add(nn.Linear(512 * 2 * 2, 1024))
model_D:add(nn.BatchNormalization(1024))
model_D:add(nn.LeakyReLU(0.2, true))

head_D = nn.Sequential()
head_D:add(nn.Linear(1024, 1))
head_D:add(nn.Sigmoid())
head_D:add(nn.View(1))

class_D = nn.Sequential()
class_D:add(nn.Linear(1024, opt.condDim))
class_D:add(nn.LogSoftMax())

discriminator = nn.Sequential()
discriminator:add(model_D)
discriminator:add(nn.ConcatTable())
discriminator:add(head_D)
discriminator:add(class_D)

----------------------------------------------------------------------
-- define Generator
model_G = nn.Sequential()
model_G:add(nn.JoinTable(2))
model_G:add(nn.Linear(opt.noiseDim + opt.condDim, 1024))
model_G:add(nn.BatchNormalization(1024))
model_G:add(nn.ReLU(true))
model_G:add(nn.Linear(1024, 512 * 2 * 2))
model_G:add(nn.BatchNormalization(512 * 2 * 2))
model_G:add(nn.ReLU(true))
model_G:add(nn.Reshape(512, 2, 2))
model_G:add(nn.SpatialFullConvolution(512, 256, 4, 4, 2, 2, 1, 1))
model_G:add(nn.SpatialBatchNormalization(256))
model_G:add(nn.ReLU(true))
model_G:add(nn.SpatialFullConvolution(256, 128, 4, 4, 2, 2, 1, 1))
model_G:add(nn.SpatialBatchNormalization(128))
model_G:add(nn.ReLU(true))
model_G:add(nn.SpatialFullConvolution(128, 64, 4, 4, 2, 2, 1, 1))
model_G:add(nn.SpatialBatchNormalization(64))
model_G:add(nn.ReLU(true))
model_G:add(nn.SpatialFullConvolution(64, 1, 4, 4, 2, 2, 1, 1))
model_G:add(nn.Sigmoid())


head_criterion = nn.BCECriterion()
class_criterion = nn.ClassNLLCriterion()


local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias = nil
      m.gradBias = nil
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

discriminator:apply(weights_init)
model_G:apply(weights_init)

params_D, grads_D = discriminator:getParameters()
params_G, grads_G = model_G:getParameters()


print('Discriminator network:')
print(discriminator)
print('Generator network:')
print(model_G)


----------------------------------------------------------------------
-- setup data
local ntrain = 50000
local nval = 5000

-- create training set and normalize
trainData = mnist.loadTrainSet(1, ntrain)
trainData:normalize()

-- create validation set and normalize
valData = mnist.loadTrainSet(ntrain+1, ntrain+nval)
valData:normalize()

if opt.gpu > 0 then
  if pcall(require, 'cudnn') then
      require 'cudnn'
      cudnn.benchmark = true
      cudnn.fastest = true
      cudnn.convert(model_G, cudnn)
      cudnn.convert(discriminator, cudnn)
   end
  discriminator:cuda()
  model_G:cuda()
  head_criterion:cuda()
  class_criterion:cuda()
end

optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}

epoch = 1

for i=1,opt.nEpochs do
  print('Epoch no:' .. epoch) 
  adversarial.train(trainData)
  adversarial.test(valData)
  adversarial.plotSamples()
  epoch = epoch + 1
end
