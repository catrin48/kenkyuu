#力場レプリカ交換と温度レプリカ交換の交換確率を求めています。
#温度はあまり気にしていないが、重要なことは力場レプリカ交換、またenkephalinは300Kで2%程度、
#chignolinはほぼ0%なのでenkephalinで計算した。chignolinもう少し計算したらアクセプタンス改善される可能性はある。
#swap_ffrem_[temp].datがffrem,swap_[temp]_[temp+1].datがtrem
#もう少し色々計算をまとめる。
inputfile="temp_sample.out"

def count1(filename):
    count0=0
    count1=0
    with open(filename,"r") as file:
     for value in file:
         value=value.strip()
         if value=="0":
           count0+=1
         elif value=="1":
           count1+=1
     total=count0+count1
     acceptance=100*count1/(count0+count1)   
     print(f"{filename}, 0の数: {count0}, １の数: {count1}, アクセプタンス: {acceptance})         
def makefilename(inputfile):
    with open(inputifle,"r") as file:
        filenamelist=[line.strip() for line in file]
    for filename in filenamelist:
        count1(f"swap_ffrem_{filename}.dat")
    for i in range(len(filenamelist)-1):
        count1(f"swap_{filenamelist[i]}_{filenamelist[i+1]}.dat")
makefilename(inputfile)        
