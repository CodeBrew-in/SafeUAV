# Demo script (5 minutes)

Start the console with `.\run.ps1`. Use **Classic** mode if no AI is set up; everything below works there except
the free-form phrasing in step 3. Press **Reset** between parts if you like.

1. **Plan before flying.** Click *Nava Nalanda Library*, then **Make target**. Click *Directorate*, **Make target**.
   Nothing moves: the banner says "Check the plan, then press Start". Open the *Mission* tab: base, targets in order, settings.
2. **Restrict a building and draw an area.** Click *E Hostel* → **Restrict**. Press **Area**, then press on the map and drag
   outward (20 to 300 m); the card that opens offers **Restrict** / **Make target** / **Delete area**. Restrict it.
3. **Start.** Press **Start** (speed 2× by default; try 5×). Watch the dashed route go around the restricted places,
   the black trail, the battery bar (the red mark is the reserve the drone never uses) and the banner (ETA, state).
4. **Change the plan in flight.** In *Chat*, type "also inspect Venture Lab". Or click a building and press **Make target**.
   A grey ghost route and a "Replanned" chip show what changed.
5. **Try an attack.** Press the preset **5 · Injection attack**, then Send. The message is refused with a reason and the drone keeps
   its plan. Try **4 · Try the jail** too. The shield counter goes up.
6. **Impossible order.** Reset, set Battery to about 25 Wh in the Mission tab, add one target and press Start. Then add four or five far-away targets. The new order is more than the battery allows, so the drone drops the old route, lands on a pad and waits ("On the ground, waiting"). Remove targets or send a smaller order and it takes off again.
7. **Take control.** While flying: **Land here** (click a pad, then the button) lands at that pad if the shield accepts it, otherwise it
   refuses and says why. **Land now** goes to the nearest pad. **Return now** goes home. **End mission** lands and finishes; then
   a new order starts a new leg.
8. **Download.** Open the *Log* tab: **Flight report** (HTML, opens in any browser), **Audit log** (JSON lines), **Track CSV**.

What to point out: nothing starts by itself; the AI only writes a small mission; the shield checks every route; the safety rules are
not something a chat message can change; the limits section in the README.
