Usage:
To utilize this product, one must get onto a non-USAFA wifi (not the ECE wifi). I used a personal hotspot. If using on my Jetson, you must first build the container with the following code in the terminal:
> sudo docker buildx build . -t weather

*Note" the container may already be running, so this step might not be necessary. 

Then, we can execute the code with this line:
> sudo docker run --network=host -it --rm --device=/dev/snd --device=/dev/gpiochip0 --runtime=nvidia --ipc=host -v huggingface:/huggingface/ weather

Once you see the line that says, "Done", you may say the key word to activate the voice prompt. Say, "pickle juice" and a line should pop up that says, "Recording.."
Ask for the weather in a certain place. It can be a city or an airport or even a geological feature like the Eiffel Tower.

It should eventually give you the weather in the requested area!


Documentation Statement: I, C2C Megan Leong, received Ei from Captain Yarbrough on three occasions for this final project. Two of them were for the system diagrams/flow diagrams, and the other was the integration of different portions of the project in main. C1C Mahajan suggested I go back to the keyword lab because there was something to be done to the arduino code. 
