function rsu7(in_ch=3, mid_ch=12, out_ch=3)
    rebnconvin = Chain(rebnconv(in_ch,out_ch,1)...)

    rebnconv1 = rebnconv_pool(out_ch, mid_ch, 1)
    rebnconv2 = rebnconv_pool(mid_ch, mid_ch, 1)
    rebnconv3 = rebnconv_pool(mid_ch, mid_ch, 1)
    rebnconv4 = rebnconv_pool(mid_ch, mid_ch, 1)
    rebnconv5 = rebnconv_pool(mid_ch, mid_ch, 1)

    rebnconv6 = Chain(rebnconv(mid_ch,mid_ch,1)...)
    rebnconv7 = Chain(rebnconv(mid_ch,mid_ch,2)...)

    rebnconv6d = rebnconv_upsample(mid_ch*2,mid_ch,1)
    rebnconv5d = rebnconv_upsample(mid_ch*2,mid_ch,1)
    rebnconv4d = rebnconv_upsample(mid_ch*2,mid_ch,1)
    rebnconv3d = rebnconv_upsample(mid_ch*2,mid_ch,1)
    rebnconv2d = rebnconv_upsample(mid_ch*2,mid_ch,1)

    rebnconv1d = Chain(rebnconv(mid_ch*2,out_ch,1)...)

    layer = x -> begin
        hxin = rebnconvin(x)

        hx1, hx = rebnconv1(hxin)
        hx2, hx = rebnconv2(hx)
        hx3, hx = rebnconv3(hx)
        hx4, hx = rebnconv4(hx)
        hx5, hx = rebnconv5(hx)

        hx6 = rebnconv6(hx)
        hx7 = rebnconv7(hx6)

        hx6dup = rebnconv6d(hx5, hx6, hx7)
        hx5dup = rebnconv5d(hx4, hx5, hx6dup)
        hx4dup = rebnconv4d(hx3, hx4, hx5dup)
        hx3dup = rebnconv3d(hx2, hx3, hx4dup)
        hx2dup = rebnconv6d(hx1, hx2, hx3dup)

        hx1d = rebnconv1d(cat(hx2dup, hx1; dims=3))

        return hx1d + hxin
    end
end

function rsu6(in_ch=3, mid_ch=12, out_ch=3)
    rebnconvin = Chain(rebnconv(in_ch,out_ch,1)...)

    rebnconv1 = rebnconv_pool(out_ch, mid_ch, 1)
    rebnconv2 = rebnconv_pool(mid_ch, mid_ch, 1)
    rebnconv3 = rebnconv_pool(mid_ch, mid_ch, 1)
    rebnconv4 = rebnconv_pool(mid_ch, mid_ch, 1)

    rebnconv5 = Chain(rebnconv(mid_ch,mid_ch,1)...)
    rebnconv6 = Chain(rebnconv(mid_ch,mid_ch,2)...)

    rebnconv5d = rebnconv_upsample(mid_ch*2,mid_ch,1)
    rebnconv4d = rebnconv_upsample(mid_ch*2,mid_ch,1)
    rebnconv3d = rebnconv_upsample(mid_ch*2,mid_ch,1)
    rebnconv2d = rebnconv_upsample(mid_ch*2,mid_ch,1)

    rebnconv1d = Chain(rebnconv(mid_ch*2,out_ch,1)...)

    layer = x -> begin
        hxin = rebnconvin(x)

        hx1, hx = rebnconv1(hxin)
        hx2, hx = rebnconv2(hx)
        hx3, hx = rebnconv3(hx)
        hx4, hx = rebnconv4(hx)

        hx5 = rebnconv5(hx)
        hx6 = rebnconv6(hx5)

        hx5dup = rebnconv5d(hx4, hx5, hx6)
        hx4dup = rebnconv4d(hx3, hx4, hx5dup)
        hx3dup = rebnconv3d(hx2, hx3, hx4dup)
        hx2dup = rebnconv2d(hx1, hx2, hx3dup)

        hx1d = rebnconv1d(cat(hx2dup, hx1; dims=3))

        return hx1d + hxin
    end
end

function rsu5(in_ch=3, mid_ch=12, out_ch=3)
    rebnconvin = Chain(rebnconv(in_ch,out_ch,1)...)

    rebnconv1 = rebnconv_pool(out_ch, mid_ch, 1)
    rebnconv2 = rebnconv_pool(mid_ch, mid_ch, 1)
    rebnconv3 = rebnconv_pool(mid_ch, mid_ch, 1)

    rebnconv4 = Chain(rebnconv(mid_ch,mid_ch,1)...)
    rebnconv5 = Chain(rebnconv(mid_ch,mid_ch,2)...)

    rebnconv4d = rebnconv_upsample(mid_ch*2,mid_ch,1)
    rebnconv3d = rebnconv_upsample(mid_ch*2,mid_ch,1)
    rebnconv2d = rebnconv_upsample(mid_ch*2,mid_ch,1)

    rebnconv1d = Chain(rebnconv(mid_ch*2,out_ch,1)...)

    layer = x -> begin
        hxin = rebnconvin(x)

        hx1, hx = rebnconv1(hxin)
        hx2, hx = rebnconv2(hx)
        hx3, hx = rebnconv3(hx)

        hx4 = rebnconv4(hx)
        hx5 = rebnconv5(hx5)

        hx4dup = rebnconv4d(hx3, hx4, hx5)
        hx3dup = rebnconv3d(hx2, hx3, hx4dup)
        hx2dup = rebnconv2d(hx1, hx2, hx3dup)

        hx1d = rebnconv1d(cat(hx2dup, hx1; dims=3))

        return hx1d + hxin
    end
end

function rsu4(in_ch=3, mid_ch=12, out_ch=3)
    rebnconvin = Chain(rebnconv(in_ch,out_ch,1)...)

    rebnconv1 = rebnconv_pool(out_ch, mid_ch, 1)
    rebnconv2 = rebnconv_pool(mid_ch, mid_ch, 1)

    rebnconv3 = Chain(rebnconv(mid_ch,mid_ch,1)...)
    rebnconv4 = Chain(rebnconv(mid_ch,mid_ch,2)...)

    rebnconv3d = rebnconv_upsample(mid_ch*2,mid_ch,1)
    rebnconv2d = rebnconv_upsample(mid_ch*2,mid_ch,1)

    rebnconv1d = Chain(rebnconv(mid_ch*2,out_ch,1)...)

    layer = x -> begin
        hxin = rebnconvin(x)

        hx1, hx = rebnconv1(hxin)
        hx2, hx = rebnconv2(hx)

        hx3 = rebnconv3(hx)
        hx4 = rebnconv4(hx3)

        hx3dup = rebnconv3d(hx2, hx3, hx4)
        hx2dup = rebnconv2d(hx1, hx2, hx3dup)

        hx1d = rebnconv1d(cat(hx2dup, hx1; dims=3))

        return hx1d + hxin
    end
end

function rsu4f(in_ch=3, mid_ch=12, out_ch=3)
    rebnconvin = Chain(rebnconv(in_ch,out_ch,1)...)

    rebnconv1 = Chain(rebnconv(out_ch,mid_ch,1)...)
    rebnconv2 = Chain(rebnconv(mid_ch,mid_ch,2)...)
    rebnconv3 = Chain(rebnconv(mid_ch,mid_ch,4)...)

    rebnconv4 = Chain(rebnconv(mid_ch,mid_ch,8)...)

    rebnconv3d = Chain(rebnconv(mid_ch*2,mid_ch,4)...)
    rebnconv2d = Chain(rebnconv(mid_ch*2,mid_ch,2)...)

    rebnconv1d = Chain(rebnconv(mid_ch*2,out_ch,1)...)

    layer = x -> begin
        hxin = rebnconvin(x)

        hx1 = rebnconv1(hxin)
        hx2 = rebnconv2(hx)
        hx3 = rebnconv3(hx)

        hx4 = rebnconv4(hx)

        hx3dup = rebnconv3d(cat(hx4, hx3; dims=3))
        hx2dup = rebnconv2d(cat(hx3d, hx2; dims=3))
        hx1dup = rebnconv1d(cat(hx2d, hx1; dims=3))

        return hx1d + hxin
    end
end

function U2Net(in_ch=3, out_ch=1)
    stage1 = rsu7(in_ch, 32, 64)
    pool12 = MaxPool((2, 2),stride=2, pad=SamePad())

    stage2 = rsu6(64, 32, 128)
    pool23 = MaxPool((2, 2),stride=2, pad=SamePad())

    stage3 = rsu5(128, 64, 512)
    pool34 = MaxPool((2, 2),stride=2, pad=SamePad())

    stage4 = rsu4(256, 128, 512)
    pool45 = MaxPool((2, 2),stride=2, pad=SamePad())

    stage5 = rsu4f(512, 256, 512)
    MaxPool((2, 2),stride=2, pad=SamePad())

    stage6 = rsu4f(512, 256, 512)

    stage5d = rsu4f(1024, 256, 512)
    stage4d = rsu4(1024, 128, 256)
    stage3d = rsu5(512, 64, 128)
    stage2d = rsu6(256, 32, 64)
    stage1d = rsu7(128, 16, 64)

    side1 = Conv((3, 3), 64 => out_ch; pad=1)
    side2 = Conv((3, 3), 64 => out_ch; pad=1)
    side3 = Conv((3, 3), 128 => out_ch; pad=1)
    side4 = Conv((3, 3), 256 => out_ch; pad=1)
    side5 = Conv((3, 3), 512 => out_ch; pad=1)
    side6 = Conv((3, 3), 512 => out_ch; pad=1)

    outconv = side1 = Conv((1, 1), 6 => out_ch)

    layer = x -> begin
        hx1 = stage1(x)
        hx = pool12(hx1)

        hx2 = stage2(hx)
        hx = pool23(hx2)

        hx3 = stage3(hx)
        hx = pool34(hx3)

        hx4 = stage4(hx)
        hx = pool45(hx4)

        hx5 = stage5(hx)
        hx = pool56(hx5)

        hx6 = stage6(hx)
        hx6up = upsample(hx6,hx5)

        hx5d = stage5d(cat(hx6up,hx5; dims=3))
        hx5dup = upsample(hx5d,hx4)

        hx4d = stage4d(cat(hx5up,hx4; dims=3))
        hx4dup = upsample(hx4d,hx3)

        hx3d = stage3d(cat(hx4up,hx3; dims=3))
        hx3dup = upsample(hx3d,hx2)

        hx2d = stage2d(cat(hx3up,hx2; dims=3))
        hx2dup = upsample(hx2d,hx1)

        hx1d = stage1d(cat(hx2up,hx1; dims=3))

        d1 = side1(hx1d)

        d2 = side2(hx2d)
        d2 = upsample(d2,d1)

        d3 = side3(hx3d)
        d3 = upsample(d3,d1)

        d4 = side4(hx4d)
        d4 = upsample(d4,d1)

        d5 = side5(hx5d)
        d5 = upsample(d5,d1)

        d6 = side6(hx6)
        d6 = upsample(d6,d1)

        d0 = outconv(cat(d1,d2,d3,d4,d5,d6; dims=3))

        return sigmoid(d0), sigmoid(d1), sigmoid(d2), sigmoid(d3), sigmoid(d4), sigmoid(d5), sigmoid(d6)
    end
end
