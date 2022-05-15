import string
import random
import sys
  
word = lambda len : ''.join(random.sample(string.ascii_letters, int(len)))
   
def main(argv):
    _, K, N, MAX_ITER = argv
    sub = word(random.choice(range(int(N), 2*int(N))))
    f = open("data.txt", "w")
    f.write("{} {} {}\n{}\n".format(K, N, MAX_ITER, sub))
    for i in range(int(K)**2):
        f.write("{}\n".format(word(2*int(N))))
    f.close()
    

if __name__ == '__main__':
    if(len(sys.argv) != 4):
        print("usage: python gen_data.py K N MaxIteration")
        exit()
    main(sys.argv)