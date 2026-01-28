To start analysing the different implementations, I decided to use a sine wave, rather than an actual wav file, because this way I can ensure the algorithms are working as expected. If any discrepancy is present in the output after applying the blackbox, (i.e. the sine wave is cut, not consistent), we can catch the bug early. 

# Time Domain

## Resampling - not so good
The easiest way of changing the pitch by a factor x is to resample the initial signal. The solution is to either reduce the number of samples or to add more samples through interpolation. resample_signal implements this behaviour.

## Overlap-Add
https://en.wikipedia.org/wiki/Overlap%E2%80%93add_method

The first step in the overlap-add is to divide the signal into multiple frames. This should be chosen based on a parameter called analysis hop. This determines the width of each frame that will be cut from the initial signal.
The second parameter is the synthesis hop.

explica tot procesul care face transformarea frameului prin fft si in ifft se adauga niste elemente pentru ca am facut zero padding ca sa nu am aliasing - vrem convolutie liniara nu circulara