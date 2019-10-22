# libdl

My awesome deep learning library from scratch

To build and run the code;
1. clone the repository into your computer.
2. Using terminal, go to that repository that has been just cloned. Run the following to hve hdf5 for the python --> ./hdf5_build_install.sh 
3. Create a build directory there. Later go inside this build directory by using the
cd build command.
4. In terminal, just run the following --> cmake ..
5. Later, still in the same build directory run --> make
6. Following that just type this line to go to source directory --> cd ../src 
7. In the source directory type the following --> python3 setup.py build_ext -i
8. Later on there is a jupyter notebook where you can actually see the visual results with pretrained weights.
9. Run jupyter-notebook or jupyter notebook, this way you will be able to reach the jupyter file.
10. To be able to work fine with jupyter notebook there are 2 things to keep in mind, you can change the input sizes, by using ../Malaria_Dataset/data_28.hdf5 or ../Malaria_Dataset/data_64.hdf5, so it means there are different weight files for different network types and the different image sizes. So if you wan to test with 64x64x3 images then you should first change the datapath to data_64.hdf5 follow the titles in the jupyter notebook and test with the part for 64x64x3 images. There is also two different network definitions. You should run one of them and then test it.
11. To create a doxygen, you need to run following from the libdl repository --> cd docs
12. In the docs directory run the following --> doxygen Doxyfile
13. Above line will generate a build directory, run the following --> cd build/html
14. From the html directory by running the following you will see the doxygen --> firefox index.html

