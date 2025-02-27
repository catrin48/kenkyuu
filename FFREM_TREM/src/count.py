inputfile="temp_sample.out"

def count1(filename):
    with open(filename,"r") as file:
     for value in file:
         value=value.strip()
         if value=="0":

def makefilename(inputifile):
    with open(inputifle,"r") as file:
        filename=[line.strip() for line in file]
