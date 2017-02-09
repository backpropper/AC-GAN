local adv = {}

local input = torch.Tensor(opt.batchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
local class = torch.Tensor(opt.batchSize, opt.condDim)
local fake_class = torch.Tensor(opt.batchSize, opt.condDim)
local target = torch.Tensor(2 * opt.batchSize)
local noise = torch.Tensor(opt.batchSize, opt.noiseDim)
local gen_noise = torch.Tensor(2 * opt.batchSize, opt.noiseDim)
local gen_class = torch.Tensor(2 * opt.batchSize, opt.condDim)
local num_class = torch.Tensor(opt.batchSize)
local real_class = torch.Tensor(opt.batchSize)
local gennum_class = torch.Tensor(2 * opt.batchSize)


if opt.gpu > 0 then
    input = input:cuda()
    class = class:cuda()
    noise = noise:cuda()
    fake_class = fake_class:cuda()
    real_class = real_class:cuda()
    num_class = num_class:cuda()
    target = target:cuda()
    gen_noise = gen_noise:cuda()
    gen_class = gen_class:cuda()
    gennum_class = gennum_class:cuda()
end


local function updateD(x)
  grads_D:zero()
  grads_G:zero()

  trainData:getBatch(input, class)
  if opt.noise == 'uniform' then
      noise:uniform(-1, 1)
  elseif opt.noise == 'normal' then
      noise:normal(0, 1)
  end
  
  if opt.gpu > 0 then
      num_class = num_class:float():random(1, 10)
      num_class = num_class:cuda()
  else
      num_class:random(1, 10)
  end
  
  for i=1,opt.batchSize do
      fake_class[i][num_class[i]] = 1
      for j=1,10 do
          if class[i][j] == 1 then
              real_class[i] = j
              break
          end
      end
  end

  target:sub(1, opt.batchSize):fill(1)

  local fake_input = model_G:forward({noise, fake_class})
  local combined_input = torch.cat(input, fake_input, 1)
  local combined_class = torch.cat(real_class, num_class, 1)

  local body_output = model_D:forward(combined_input)
  local head_output = head_D:forward(body_output)
  local class_output = class_D:forward(body_output)
  
  local head_loss = head_criterion:forward(head_output, target)
  local dhead_do = head_criterion:backward(head_output, target)
  local head_grad = head_D:backward(body_output, dhead_do)

  local class_loss = class_criterion:forward(class_output, combined_class)
  local dclass_do = class_criterion:backward(class_output, combined_class)
  local class_grad = class_D:backward(body_output, dclass_do)

  model_D:backward(combined_input, head_grad + class_grad)
  
  loss = head_loss + class_loss  
  return loss, grads_D
end

local function updateG(x)
  grads_D:zero()
  grads_G:zero()

  if opt.noise == 'uniform' then
      gen_noise:uniform(-1, 1)
  elseif opt.noise == 'normal' then
      gen_noise:normal(0, 1)
  end

  if opt.gpu > 0 then
      gennum_class = gennum_class:float():random(1, 10)
      gennum_class = gennum_class:cuda()
  else
      gennum_class = gennum_class:random(1, 10)
  end

  for i=1,2*opt.batchSize do
      gen_class[i][gennum_class[i]] = 1
  end
  
  target:fill(1)  

  local fake_input = model_G:forward({gen_noise, gen_class})
  local body_output = model_D:forward(fake_input)
  local head_output = head_D:forward(body_output)
  local class_output = class_D:forward(body_output)

  local head_loss = head_criterion:forward(head_output, target)
  local dhead_do = head_criterion:backward(head_output, target)
  local head_grad = head_D:updateGradInput(body_output, dhead_do)

  local class_loss = class_criterion:forward(class_output, gennum_class)
  local dclass_do = class_criterion:backward(class_output, gennum_class)
  local class_grad = class_D:updateGradInput(body_output, dclass_do)
  
  local disc_grad = model_D:updateGradInput(fake_input, head_grad + class_grad)
  model_G:backward({gen_noise, gen_class}, disc_grad)

  loss = head_loss + class_loss
  return loss, grads_G
end

function adv.train(dataset)
  print('Training')
  for i=1,dataset:size(),opt.batchSize do
      xlua.progress(i,dataset:size())
      optim.adam(updateD, params_D, optimStateD)
      optim.adam(updateG, params_G, optimStateG)
  end
end

function adv.test(dataset)
  print('Testing')
  totaldiscloss = 0
  totalgenloss = 0
  batches = 0

  for ep=1,dataset:size(),opt.batchSize do
    xlua.progress(ep, dataset:size())
    batches = batches + 1
    dataset:getBatch(input, class)

    if opt.noise == 'uniform' then
        noise:uniform(-1, 1)
    elseif opt.noise == 'normal' then
        noise:normal(0, 1)
    end

    if opt.gpu > 0 then
        num_class = num_class:float():random(1, 10)
        num_class = num_class:cuda()
    else
        num_class:random(1, 10)
    end

    for i=1,opt.batchSize do
        fake_class[i][num_class[i]] = 1
        for j=1,10 do
            if class[i][j] == 1 then
                real_class[i] = j
                break
            end
        end
    end

    target:sub(1, opt.batchSize):fill(1)

    local fake_input = model_G:forward({noise, fake_class})
    local combined_input = torch.cat(input, fake_input, 1)
    local combined_class = torch.cat(real_class, num_class, 1)

    local body_output = model_D:forward(combined_input)
    local head_output = head_D:forward(body_output)
    local class_output = class_D:forward(body_output)

    local head_loss = head_criterion:forward(head_output, target)
    local class_loss = class_criterion:forward(class_output, combined_class)

    disc_loss = head_loss + class_loss

    if opt.noise == 'uniform' then
        gen_noise:uniform(-1, 1)
    elseif opt.noise == 'normal' then
        gen_noise:normal(0, 1)
    end

    if opt.gpu > 0 then
        gennum_class = gennum_class:float():random(1, 10)
        gennum_class = gennum_class:cuda()
    else
        gennum_class:random(1, 10)
    end

    for i=1,2*opt.batchSize do
        gen_class[i][gennum_class[i]] = 1
    end
    
    target:fill(1)  

    local fake_input = model_G:forward({gen_noise, gen_class})
    local body_output = model_D:forward(fake_input)
    local head_output = head_D:forward(body_output)
    local class_output = class_D:forward(body_output)

    local head_loss = head_criterion:forward(head_output, target)
    local class_loss = class_criterion:forward(class_output, gennum_class)

    gen_loss = head_loss + class_loss

    totaldiscloss = totaldiscloss + disc_loss
    totalgenloss = totalgenloss + gen_loss
  end

  print("Disriminator Loss: " .. totaldiscloss / batches)
  print("Generator Loss: " .. totalgenloss / batches)

end

function adv.plotSamples()
  local class = 1
  local fake_class = torch.zeros(100, opt.condDim)
  for i = 1,100 do
      fake_class[i][class] = 1
      if i % 10 == 0 then class = class + 1 end
  end

  local noise = torch.Tensor(100, opt.noiseDim)
  if opt.noise == 'uniform' then
      noise:uniform(-1, 1)
  elseif opt.noise == 'normal' then
      noise:normal(0, 1)
  end
  local x_gen = model_G:forward({noise, fake_class})
  local to_plot = {}
  for n = 1,100 do
      to_plot[n] = x_gen[n]:float()
  end

  paths.mkdir('samples')
  local fname = paths.concat('samples/' .. epoch .. '.png')

  image.save(fname, image.toDisplayTensor{input=to_plot, scaleeach=true, nrow=10})

  paths.mkdir('checkpoints')
  torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_generator.t7', model_G:clearState())
  torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_discriminator.t7', discriminator:clearState())

end

return adv
