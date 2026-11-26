import argparse
import itk
import trimesh

parser = argparse.ArgumentParser(description = 'CT reconstruction')

parser.add_argument('-in', '--input_dir', default = '', help='Input directory for PNG files')
parser.add_argument('-out', '--output_dir', default = '', help='Output directory for RAW file')


args = parser.parse_args()

inputFileName = args.input_dir
outputFileName = args.output_dir

Dimension = 3
PixelType = itk.UC
ImageType = itk.Image[PixelType, Dimension]

ReaderType = itk.ImageFileReader[ImageType]
reader = ReaderType.New()
reader.SetFileName(inputFileName)


# ######################
# DoubleImageType = itk.Image[itk.D, Dimension]
# CastToDoubleFilterType = itk.CastImageFilter[ImageType, DoubleImageType] 
# castToDouble = CastToDoubleFilterType.New()
# castToDouble.SetInput(reader.GetOutput())

# AntiAliasFilterType =  itk.AntiAliasBinaryImageFilter[DoubleImageType, DoubleImageType]
# antiAliasFilter = AntiAliasFilterType.New()
# antiAliasFilter.SetMaximumRMSError(0.07)
# antiAliasFilter.SetNumberOfIterations(30)
# antiAliasFilter.SetNumberOfLayers(2)
# antiAliasFilter.SetInput(castToDouble.GetOutput())

# CastToUCFilterType = itk.CastImageFilter[DoubleImageType,ImageType] 
# castToUC = CastToUCFilterType.New()
# castToUC.SetInput(antiAliasFilter.GetOutput())

# #######################


# the range of number that will be catched (I guess ) 
lowerThreshold = 255
upperThreshold =  255

# Set the threshold
BinaryThresholdFilterType = itk.BinaryThresholdImageFilter[ImageType, ImageType]
threshold = BinaryThresholdFilterType.New()

threshold.SetInput(reader.GetOutput())
threshold.SetLowerThreshold(lowerThreshold)
threshold.SetUpperThreshold(upperThreshold)
threshold.SetOutsideValue(0)


MeshType = itk.Mesh[itk.D,Dimension]
#transform
# matrx = [[0.226, 0,0,0],[0,0.226, 0, 0],[0, 0 ,0.226,0],[0 ,0, 0 ,1]]
# MeshType.apply_transform(matrx)

#
FilterType = itk.BinaryMask3DMeshSource[ImageType, MeshType]
filter = FilterType.New()
filter.SetInput(threshold.GetOutput())
filter.SetObjectValue(255)
filter.Update

# Output the mesh
WriterType = itk.MeshFileWriter[MeshType]
writer = WriterType.New()
writer.SetFileName(outputFileName)
writer.SetInput(filter.GetOutput())

writer.Update()

