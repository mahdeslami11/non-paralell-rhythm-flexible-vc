# Non-parallel-rhythm-flexible-VC
PyTorch implementation of: 
[Rhythm-Flexible Voice Conversion without Parallel Data Using Cycle-GAN over Phoneme Posteriorgram Sequences](https://arxiv.org/abs/1808.03113)

* This repo is NOT completed yet
* This repo is NOT completed yet
* This repo is NOT completed yet
* Please new issues if you find something werid or not working, thanks!

## Samples
Samples could be found [here](./samples), the corresponding experiment is specified at section 5.3 in the paper. Only conventional and proposed methods are compared here.

## Python and Toolkit Version
    Python:   '3.5.2'
    Numpy:    '1.16.2'
    PyTorch:  '0.4.1'
    Montreal-force-aligner: '1.1.0'

## Data Preprocess (Frame-level phoneme boundary segmentation included)
1. Download and decompress VCTK corpus
2. Put text file and audio file under same dir, run `rename.sh`
3. Run align_VCTK.sh to get aligned result
4. Set path info in config/config.yaml
5. Run `preprocess.py` to generate acoustic features with corresponding phone label

## Configuration and Usage
1. All hyperparameters are listed in [this .yaml file](./config/config.yaml)
2. All modules training could be done by calling the `main.py` by adding different arguments. 
<pre><code>
usage: main.py [-h] [--config CONFIG] 
               [--seed SEED] [--train | --test]
               [--ppr | --ppts | --uppt] 
               [--spk_id SPK_ID] [--A_id A_ID] [--B_id B_ID] 
               [--pre_train]
</code></pre>
3. The detailed usages of each module are listed below.
4. The path of logging and model saving should be specified in config file first.

### PPR
[Example script](./ppr.sh)
### Training
<pre><code>python3 main.py --config [path-to-config] --train --ppr</code></pre>
### Evaluation
<pre><code>python3 main.py --config [path-to-config] --test --ppr</code></pre>

## PPTS
[Example script](./ppts.sh)
### Training
<pre><code>python3 main.py --config [path-to-config] --train --ppts \\
                --spk_id [which-speaker-to-train]</code></pre>

### Evaluation
<pre><code>python3 main.py --config [path-to-config] --test --ppts \\
                --spk_id [which-speaker-to-train]</code></pre>

## UPPT(CycleGAN ver.)
[Example script](./uppt.sh)
### AE Pre-Training
<pre><code>python3 main.py --config [path-to-config] --train --uppt \\
    --pre_train --A_id [src-speaker] --B_id [tgt-speaker]</code></pre>
* If A_id and B_id are both set to "all", then data of two groups of fast and slow speakers instead of two single speaker will be used instead for pre-training.
* Ex. <pre><code> ... --A_id all --B_id all</code></pre>

### Training
<pre><code>python3 main.py --config [path-to-config] --train --uppt \\
    --A_id [src-speaker] --B_id [tgt-speaker]</code></pre>

### Evaluation
<pre><code>python3 main.py --config [path-to-config] --test --uppt \\
    --A_id [src-speaker] --B_id [tgt-speaker]</code></pre>

## UPPT(StarGAN ver.)
[Example script](./star.sh)
### AE Pre-Training
<pre><code>python3 star_main.py 
--config [path-to-config] --train --uppt --pre_train</code></pre>

### Training
<pre><code>python3 star_main.py --config [path-to-config] --train --uppt</code></pre>

### Evaluation
<pre><code>python3 star_main.py --config [path-to-config] --test --uppt \\
    --tgt_id [tgt-speaker]</code></pre>


## Notes
1. Phoneme 'spn' means Unknown in MFA, so currently map it with 'sp' to id 0 as well.
2. Is padding 'sp' a good choice? Or maybe 'sil'?

## TODO
- [x] Add Logging method to solver, removing add summ redundancy in both train and eval
- [ ] Whole conversion process pipeline, adding functions to load from specified path at inference time
- [ ] StarGAN inference
‏var SpeechRecognition = SpeechRecognition || webkitSpeechRecognition
‏var recognition = new SpeechRecognition();
recognition.continuous = false;
recognition.lang = 'en-US';
recognition.interimResults = false;
recognition.maxAlternatives = 1;
recognition.start();
recognition.stop();
recognition.onresult = function(event) {
  var result = event.results[0][0].transcript;
}
recognition.onnomatch = function(event) {}
recognition.onerror= function(event) {}

<!DOCTYPE html>
<html lang="fa">
<head>
    <meta charset="UTF-8">
    <title>Rapidcode.iR - سورس کد</title>
    <link rel="stylesheet" href="static/css/main.css">
    <link rel="stylesheet" href="static/css/lib/normalize.css">
    <link rel="stylesheet" href="static/css/lib/skeleton.css">
    <link rel="stylesheet" href="static/css/lib/persian-datepicker.min.css">
</head>
<body>
 <script src="static/js/lib/jquery-3.2.1.min.js"></script>
    <script src="static/js/lib/persian-date.min.js"></script>
    <script src="static/js/lib/persian-datepicker.min.js"></script>
    <script src="static/js/app.js"></script>
    <script>
        const datepickerDOM = $("#leavingDate");
        window.dateObject = datepickerDOM.persianDatepicker(
        {
            "inline": false,
            "format": "LLLL",
            "viewMode": "day",
            "initialValue": true,
            "onSelect" : function(){ 
                const currentDateState = date.dateObject.State.gregorian;
             
                window.selectedDate = ${currentDateState.year}-${(currentDateState.month + 1).toString().padStart(2, "0")}-${currentDateState.day.toString().padStart(2, "0")};
                 
                dateObject.hide();
                getTheVoiceResult(selectedDate);
            }
        });
         
        const date = dateObject.getState().view;
    </script>
</body>
</html>   
