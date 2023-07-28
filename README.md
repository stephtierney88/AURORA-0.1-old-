# AURORA
AI-driven User-friendly Responsive Online Recreational Assistant  || Automated Universal Recreation and Online Responsive AI.

Parser <--> initprompt.txt Start Loop:

ChatGPT  <-->TextResponse <-->  Parser App<-->  (Parser app executes commands on) PC; 
PC<-->Screenshots <--> Parser  <-->  App sends Screenshots or text via api <--> ChatGPT;  

if tokenlimit>X||CMD:HANDOFF then generate handoff.txt, else Loop;

//Basically it's inspired by twitchPlaysPokemon. However the app instead reads chatGPTs chat responses then executes keyboard & mouse inputs. Timestamped screenshots give it visual context at an interval.
