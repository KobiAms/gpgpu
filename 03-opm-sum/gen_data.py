import string
import random
import sys
  
   
def main(argv):
    _, N = argv
    f = open("data.txt", "w")
    f.write("{}\n".format(N))
    for i in range(int(N)):
        f.write("{}\n".format(random.uniform(-45, 45)))
    f.close()
    

if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print("usage: python gen_data.py N")
        exit()
    main(sys.argv)