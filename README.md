# 10x10 Grid

Full observation is given.  

**Training:** Domain Randomization  
**Trained for:** 20M steps  

---

## 1. 10 Blocks
- **Mean reward over 100 episodes:** 0.8560039062499999  
- **Solved percentage:** 89%  

![10_blocks_1](Notes/10_blocks_1.gif)  
![10_blocks_2](Notes/10_blocks_2.gif)  
![10_blocks_4](Notes/10_blocks_4.gif)  
![10_blocks_3](Notes/10_blocks_3.gif)  

---

## 2. 2 Rooms
- **Mean reward over 100 episodes:** 0.8495546875  
- **Solved percentage:** 88%  

![two_room_run](Notes/two_room_run.gif)  

---

## 3. Empty Room (Out-of-Distribution)
Here the environment without blocks is never seen by the agent, so it can be considered as an out-of-distribution test.  

- **Mean reward:** 0.8433515625  
- **Solved percentage:** 87%  

![out_of_dist_1](Notes/out_of_dist_1.gif)  
![out_of_dist_2](Notes/out_of_dist_2.gif)  

---

## 4. Four Rooms
- **Mean reward:** 0.712578125  
- **Solved percentage:** 74%  

![four_room_run](Notes/four_room_run.gif)  
![four_room_run_solved](Notes/four_room_run_solved.gif)  
![four_room_run_failed](Notes/four_room_run_failed.gif)  
![four_room_run_2](Notes/four_room_run_2.gif)  

---

## 5. Labyrinth
- **Mean reward:** 0  
- **Solved percentage:** 0  

![mingle_mingle](Notes/mingle_mingle.gif)
