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

--[[
local image_list = './data/image_list.txt'

local csv = csvigo.load({path = image_list, verbose = false, mode = "raw"}) ------ csv : comma-separated values
local x = {}
for i = 1, #csv do ------ #csv : 9999
  local filename = csv[i][1] ------ filename = /CelebA/Img/img_align_celeba/Img/000755.jpg
  local img = Image.load(filename, 3, 'byte')
  img = Image.rgb2y(img)
  img = Image.scale(img, 64, 64)
  Image.saveJPG("./image/celeba-" .. i .. ".jpg", img)
end

print("image save end")
]]

local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(4, "folder", {})
print("Dataset: " .. folder, " Size: ", data:size())
---- sample output on cmd : Dataset: folder  Size:  202599





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
cutorch.setDevice(1)
input = input:cuda();  noise = noise:cuda();  label = label:cuda()

if pcall(require, 'cudnn') then
  require 'cudnn'
  cudnn.benchmark = true
  cudnn.convert(netG, cudnn)
  cudnn.convert(netD, cudnn)
end
netD:cuda();           netG:cuda();           criterion:cuda()

print("cuda end")





local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

noise_vis = noise:clone()
noise_vis:normal(0, 1)

print("noise end")



-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
  gradParametersD:zero()

  -- train with real
  data_tm:reset(); data_tm:resume()
  local real = 1 -- data:getBatch()
  data_tm:stop()
  input:copy(real)
  label:fill(real_label)

  local output = netD:forward(input)
  local errD_real = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  netD:backward(input, df_do)

  -- train with fake
  if opt.noise == 'uniform' then -- regenerate random noise
    noise:uniform(-1, 1)
  elseif opt.noise == 'normal' then
    noise:normal(0, 1)
  end
  local fake = netG:forward(noise)
  input:copy(fake)
  label:fill(fake_label)

  local output = netD:forward(input)
  local errD_fake = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  netD:backward(input, df_do)

  errD = errD_real + errD_fake

  return errD, gradParametersD
end

print("fDx end")

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
  gradParametersG:zero()

  --[[ the three lines below were already executed in fDx, so save computation
  noise:uniform(-1, 1) -- regenerate random noise
  local fake = netG:forward(noise)
  input:copy(fake) ]]--
  label:fill(real_label) -- fake labels are real for generator cost

  local output = netD.output -- netD:forward(input) was already executed in fDx, so save computation
  errG = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  local df_dg = netD:updateGradInput(input, df_do)

  netG:backward(noise, df_dg)
  return errG, gradParametersG
end

print("fGx end")





-- train
local niter = 2
local ntrain = 10000
local name = jgravity_test
for epoch = 1, niter do
   epoch_tm:reset()
   local counter = 0
   for i = 1, math.min(#x, ntrain), batchSize do
      tm:reset()
      -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      optim.adam(fDx, parametersD, optimStateD)

      -- (2) Update G network: maximize log(D(G(z)))
      optim.adam(fGx, parametersG, optimStateG)

      -- logging
      if ((i-1) / batchSize) % 1 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                   .. '  Err_G: %.4f  Err_D: %.4f'):format(
                 epoch, ((i-1) / batchSize),
                 math.floor(math.min(#x, ntrain) / batchSize),
                 tm:time().real, data_tm:time().real,
                 errG and errG or -1, errD and errD or -1))
         ---- sample output on cmd : Epoch: [1][       0 /     1012]   Time: 2.744  DataTime: 0.001    Err_G: 0.6021  Err_D: 1.9472 ...
      end
   end
   paths.mkdir('checkpoints')
   parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
   parametersG, gradParametersG = nil, nil
   torch.save('checkpoints/' .. name .. '_' .. epoch .. '_net_G.t7', netG:clearState())
   torch.save('checkpoints/' .. name .. '_' .. epoch .. '_net_D.t7', netD:clearState())
   parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
   parametersG, gradParametersG = netG:getParameters()
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, niter, epoch_tm:time().real))
   ---- sample output on cmd : End of epoch 1 / 25   Time Taken: 465.928
end
