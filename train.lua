require 'pl'
------ require file path default setting
local __FILE__ = (function() return string.gsub(debug.getinfo(2, 'S').source, "^@", "") end)()
package.path = path.join(path.dirname(__FILE__), "lib", "?.lua;") .. package.path

------ for train
require 'optim'
require 'xlua'
require 'w2nn'
------ for convert_data
local Image = require 'image'

------ for train
local settings = require 'settings'
local srcnn = require 'srcnn'
local minibatch_adam = require 'minibatch_adam'
local iproc = require 'iproc'
local reconstruct = require 'reconstruct'
local compression = require 'compression'
local pairwise_transform = require 'pairwise_transform'
local image_loader = require 'image_loader'
------ for convert_data
local cjson = require 'cjson'
local csvigo = require 'csvigo'
local alpha_util = require 'alpha_util'



require 'torch'
require 'nn'
require 'optim'

require 'cunn'

local image_list = './data/image_list.txt'

local csv = csvigo.load({path = image_list, verbose = false, mode = "raw"}) ------ csv : comma-separated values
local x = {}
for i = 1, #csv do ------ #csv : 9999
  local filename = csv[i][1] ------ filename = /CelebA/Img/img_align_celeba/Img/000755.jpg
  local img = Image.load(filename, 3, 'byte')
  img = Image.rgb2y(img)
  img = Image.scale(img, 64, 64)
  table.insert(x, {compression.compress(img), {data = {filters = filters}}})
  xlua.progress(i, #csv)
  if i % 10 == 0 then
    collectgarbage()
  end
end

print("image load end")





local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m:noBias()
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

local ngf = 64
local ndf = 64
local nz = 100
local ch = 1

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution

local netG = nn.Sequential()
netG:add(SpatialConvolution(ch, 16, 3, 3, 1, 1, 0, 0))
netG:add(nn.LeakyReLU(0.1, true))
netG:add(SpatialConvolution(16, 32, 3, 3, 1, 1, 0, 0))
netG:add(nn.LeakyReLU(0.1, true))
netG:add(SpatialConvolution(32, 64, 3, 3, 1, 1, 0, 0))
netG:add(nn.LeakyReLU(0.1, true))
netG:add(SpatialFullConvolution(64, ch, 4, 4, 2, 2, 3, 3):noBias())
netG:add(nn.View(-1):setNumInputDims(3))

netG:apply(weights_init)

print ("netG end")

local netD = nn.Sequential()

-- input is (ch) x 64 x 64
netD:add(SpatialConvolution(ch, ndf, 4, 4, 2, 2, 1, 1))
netD:add(nn.LeakyReLU(0.2, true))
-- state size: (ndf) x 32 x 32
netD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*2) x 16 x 16
netD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*4) x 8 x 8
netD:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*8) x 4 x 4
netD:add(SpatialConvolution(ndf * 8, 1, 4, 4))
netD:add(nn.Sigmoid())
-- state size: 1 x 1 x 1
netD:add(nn.View(1):setNumInputDims(3))
-- state size: 1

netD:apply(weights_init)

print ("netD end")





local function create_criterion(model)
  local offset = 1
  local output_w = settings.crop_size - offset * 2
  local weight = torch.Tensor(1, output_w * output_w)
  weight[1]:fill(1.0)
  return w2nn.ClippedWeightedHuberCriterion(weight, 0.1, {0.0, 1.0}):cuda()
end

local criterion = create_criterion(netG)

print("create_criterion end")
---------------------------------------------------------------------------
optimStateG = {
   learningRate = 0.0002,
   beta1 = 0.5,
}
optimStateD = {
   learningRate = 0.0002,
   beta1 = 0.5,
}
----------------------------------------------------------------------------
local batchSize = 200
local fineSize = 64
local input = torch.Tensor(batchSize, 3, fineSize, fineSize)
local noise = torch.Tensor(batchSize, nz, 1, 1)
local label = torch.Tensor(batchSize)
local errD, errG
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------
cutorch.setDevice(opt.gpu)
input = input:cuda();  noise = noise:cuda();  label = label:cuda()

if pcall(require, 'cudnn') then
  require 'cudnn'
  cudnn.benchmark = true
  cudnn.convert(netG, cudnn)
  cudnn.convert(netD, cudnn)
end
netD:cuda();           netG:cuda();           criterion:cuda()

print("cuda end")
