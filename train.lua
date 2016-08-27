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
