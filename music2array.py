import wave
import numpy as np

def audio2array(aud):
    f = wave.open(aud)
    params = f.getparams()
    nchannels, samplewidth, framerate, nframes = params[:4]
    str_audio = f.readframes(nframes)
    f.close()
    assert(samplewidth in [2,3,4])
    if samplewidth == 2:
        audio = np.frombuffer(str_audio,dtype=np.short)
    elif samplewidth == 3:
        audio = np.frombuffer(str_audio,dtype=np.long)
    else:
        import wavio
        audio = wavio.read(aud)
    audio = audio.reshape(-1)
    return audio, nchannels, samplewidth, framerate, nframes

def array2audio(arr, nchannels, samplewidth, framerate, nframes, wavname):

    f = wave.open(wavname,"wb")
    f.setnchannels(nchannels)
    f.setsampwidth(samplewidth)
    f.setframerate(framerate)
    f.setnframes(nframes)
    f.writeframes(arr.tostring())
    f.close()
    return f
    

#test
if __name__ == "__main__":

    arr, nchannels, samplewidth, framerate, nframes = audio2array("track1.wav")
    array2audio(arr, nchannels, samplewidth, framerate, nframes, wavname = "track1new.wav")
    arr1, nchannels1, samplewidth1, framerate1, nframes1 = audio2array("track1new.wav")
    print(arr == arr1)
    print(nchannels == nchannels1)
    print(samplewidth == samplewidth1)
    print(framerate == framerate1)
    print(nframes == nframes1)