# Ta thickness fit

A small web page that fits a tantalum film's thickness from an X-ray reflectivity scan (`.xrdml`).
Upload the scan, enter the Ta sputter power and time, press **Fit**.

The stack is Ru cap (21.4 Å, locked) / Ta (fitted) / Ru seed (21.4 Å, locked) on glass.

The fit runs entirely in your browser (Python + numpy + scipy via [Pyodide](https://pyodide.org)).
Your scan is not uploaded anywhere and nothing is stored; refreshing the page clears it.
The first visit downloads about 20 MB for the Python runtime, which the browser then caches.
