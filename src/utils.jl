function upsample(input, o_size)
    h, w, c, n = size(input)
    oh, ow, _, _ = size(o_size)
    output = zeros(oh, ow, c, n)
    for i in 1:n
        output[:,:,:,1] = imresize(input[:, :, :, i], (oh, ow, c))
    end
    return output
end

function rebnconv(in_ch::Int=3, out_ch::Int=3, dirate::Int=1)
    layers = []
    push!(layers, Conv((3, 3), in_ch => out_ch; pad=1*dirate, dilation=1*dirate))
    push!(layers, BatchNorm(out_ch, relu))
    return Tuple(layers)
end

function rebnconv_pool(in_ch=3, out_ch=3, dirate=1)
    conv = Chain(rebnconv(in_ch,out_ch,dirate)...)
    pool = MaxPool((2, 2),stride=2, pad=SamePad())

    layer = x -> begin
        x_conv = conv(x)
        x_pool = pool(x_conv)
        return x_conv, x_pool
    end
end

function rebnconv_upsample(in_ch=3, out_ch=3, dirate=1)
    conv = Chain(rebnconv(in_ch,out_ch,dirate)...)

    layer = (n, n1, n2) -> begin
        hxd = conv(cat(n2, n1; dims=3))
        return upsample(hxd, n)
    end
end
