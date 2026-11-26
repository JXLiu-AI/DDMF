import itk
import argparse
import os

parser = argparse.ArgumentParser(
    description="Smooth Binary Image Before Surface Extraction."
)
parser.add_argument("--input_dir")
parser.add_argument("--output_dir")
parser.add_argument("--maximum_RMS_error",type=float,default=0.001, metavar = 'M')
parser.add_argument("--number_of_iterations",  type=int, default=50, metavar = 'M')
parser.add_argument("--number_of_layers",  type=int ,default = 2, metavar = 'M')
args = parser.parse_args()

PixelType = itk.F
OutputType = itk.UC
Dimension = 2
ImageType = itk.Image[PixelType, Dimension]
OutType = itk.Image[OutputType,Dimension]
ReaderType = itk.ImageFileReader[ImageType]
reader = ReaderType.New()
for dirName, subdirList, fileList in os.walk(args.input_dir):
    fileList = sorted(fileList)
    for i,filename in enumerate(fileList):
        
        image = os.path.join(args.input_dir,filename)
        reader.SetFileName(image)

        AntiAliasFilterType = itk.AntiAliasBinaryImageFilter[ImageType, ImageType]
        antialiasfilter = AntiAliasFilterType.New()
        antialiasfilter.SetInput(reader.GetOutput())
        antialiasfilter.SetMaximumRMSError(args.maximum_RMS_error)
        antialiasfilter.SetNumberOfIterations(args.number_of_iterations)
        antialiasfilter.SetNumberOfLayers(args.number_of_layers)

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        castImageFilter = itk.CastImageFilter[ImageType, OutType].New()
        castImageFilter.SetInput(antialiasfilter.GetOutput())

        WriterType = itk.ImageFileWriter[OutType]
        writer = WriterType.New()
        writer.SetFileName(os.path.join(args.output_dir,filename))
        writer.SetInput(castImageFilter.GetOutput())

        writer.Update()