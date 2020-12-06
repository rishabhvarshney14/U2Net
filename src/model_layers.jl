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
        hx5 = rebnconv5(hx4)

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
        hx2 = rebnconv2(hx1)
        hx3 = rebnconv3(hx2)

        hx4 = rebnconv4(hx3)

        hx3d = rebnconv3d(cat(hx4, hx3; dims=3))
        hx2d = rebnconv2d(cat(hx3d, hx2; dims=3))
        hx1d = rebnconv1d(cat(hx2d, hx1; dims=3))

        return hx1d + hxin
    end
end
