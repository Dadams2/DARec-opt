
# Running this yourself 

```bash
uv sync
```

you want to start by downloading the data. No Guarentees that the links still work:

```bash
./data/download.sh
```

Then you want to run the preprocessing to generate the correct numpy matricies. You can run the original DARec code if you would like using the corresponding `Data_Preprocessing.py` or you can trust I implemented the multi-mode version and it all works with:


```bash
python DArec_opt/Data_Preprocessing.py
```
