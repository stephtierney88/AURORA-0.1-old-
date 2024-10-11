# AURORA
Autonomous Universal Recreation and Online Responsive AI

Parser <--> initprompt.txt Start Loop:

Loop:

ChatGPT <-->TextResponse <--> Parser App<--> (Parser app executes commands on) PC;

PC<-->Screenshots <--> Parser <--> App sends Screenshots or text via api <--> ChatGPT;


if tokenlimit>X||CMD:HANDOFF then generate handoff.txt, else Loop;

//Basically it's inspired by twitchPlaysPokemon. However the app instead reads chatGPTs chat responses then executes keyboard & mouse inputs. Timestamped screenshots give it visual context at an interval.


All you need are the files Aurora.txt and aurora.py in the AURORA folder. 
The program assumes that the init.txt file is located in the current working directory where the script is executed. It does not specify a particular folder path, so it looks for "Aurora_init.txt" (or the equivalent) directly in the directory from which the program is run.

*𝑵𝑶𝑻𝑬:* expect the AURORA-old repo to be updated infrequently
Newer updates / more frequent updates for the AURORA repo, while periodically pushing updates to AURORA-old...
