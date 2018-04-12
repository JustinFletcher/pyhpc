from __future__ import print_function

import tensorflow as tf

def main(args):
    with tf.Session() as sess:
        for device in sess.list_devices():
            print(device)

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
