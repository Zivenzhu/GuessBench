# GuessBench

This repository contains the questions and answers for GuessBench. At present, both the data and the test code are available. We welcome everyone to use and explore our benchmark!


## Introduction

GuessBench is a novel benchmark that evaluates Vision Language Models (VLMs) on modeling the pervasive, noisy, and pluralistic human creativity. GuessBench sources data from "Guess the Build", an online multiplayer Minecraft minigame where one player constructs a Minecraft build given a concept (e.g., caterpillar) and others try to guess it with natural language hints, presenting a pristine testbed for sensemaking creativity in the wild with VLMs acting as guessers. We curate 1500 images from the actual gameplay and design 2000 problems spanning static and dynamic image settings, natural language hints of varying completeness, and more. 

## Overview

![image](https://github.com/user-attachments/assets/b70adba7-bd44-413f-ae0b-638dde5509ce)

## Evaluation
```
cd "evaluation code"
python evaluation.py
```


