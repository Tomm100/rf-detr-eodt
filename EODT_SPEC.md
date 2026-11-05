# EoDT — Specifica di implementazione

> **Stato:** aggiornata il 04/11/2026 con tutte le decisioni prese. Le voci **[DA CONFERMARE]** hanno un default scelto ma non ancora confermato; le voci **[DA DECIDERE]** non hanno ancora una scelta. L'agente non cambia né le une né le altre di propria iniziativa. Tutto ciò che varia tra gli esperimenti si imposta da file di configurazione (§3.7).

## 0. Riepilogo delle decisioni

| Tema | Decisione | Stato | Dove |
|---|---|---|---|
| Backbone | DINOv2 da timm, attention globale, patch 16 al caricamento | deciso | §3.1 |
| Registri | DINOv2 con 4 registri, in EoDT e nella baseline | deciso (Niccolò) | §3.1 |
| Risoluzione | 576 px (come la Medium), uguale per tutte le taglie, configurabile | deciso | §3.1 |
| Projector | rimosso | deciso | §1 |
| Decoder | rimosso: query negli ultimi L2 blocchi del ViT | deciso | §3.2, §3.4 |
| L2 | 3 per ViT-S e ViT-B, 4 per ViT-L (come EoMT); **obbligatorio in ogni YAML**, nessun default nel codice | deciso, da ablare | §3.2 |
| Query | due modalità, scelte da configurazione. **Modello principale: `selection`** (stesso tipo di query della baseline RF-DETR, così l'ultimo anello della catena isola il solo decoder). **`learned` (come EoMT) è un'ablazione** | deciso | §3.3 |
| Numero di query | Q = 300, configurabile | deciso | §3.3 |
| Box | `learned`: box assoluto; `selection`: Δ sempre rispetto al riferimento (b0 + offset `refpoint_embed`) | deciso | §3.5 |
| Testa | una sola, condivisa da tutti i blocchi L2; prima della testa la norm finale pre-addestrata del ViT (`vit.norm`), come EoMT | deciso | §3.5 |
| Group-DETR | due alternative a confronto: (a) niente gruppi, che è anche la prima versione; (b) gruppi con patch isolati | deciso (Niccolò) | §4 |
| Baseline | RF-DETR senza Group-DETR, configurazione Medium sul ViT-S (576 px, 4 layer di decoder, 300 query); decoder scritto nel YAML | deciso | §5 |
| Catena di ablazione | con finestre → senza finestre → senza projector → EoDT, tutte riaddestrate | deciso (Niccolò) | §5 |
| Scaling a ViT-B e ViT-L | come scalare la baseline | [DA DECIDERE]; il codice deve già permetterlo da configurazione | §3.6 |
| Ricetta di training | quella di RF-DETR con warm-up e learning rate che scende; rifinita sul ViT-S e identica per EoDT e baseline | deciso | §7 |
| Numero di epoch | 12 per le prime run, uguale per EoDT e baseline; definitivo dopo le curve | partenza | §7.1 |
| Warm-up | lineare di RF-DETR; quello in due fasi di EoMT come opzione, spenta di default | da rifinire | §7.1 |
| Batch totale | 16 (4 per GPU × 4 GPU), uguale per EoDT e baseline | partenza | §7.1 |
| Checkpoint riportato | proposta: modello finale con EMA; stessa regola per tutti | [DA DECIDERE] | §7.1 |
| NAS | no | deciso | §7 |
| Objects365 | no | deciso | §7 |
| Dataset | COCO 2017: train (118.287) per il training, val (5.000) per la valutazione | deciso | §6 |
| Configurazione | un file YAML per esperimento, letto da un loader in `eodt/` (non `rfdetr fit`); nessun valore scritto nel codice | deciso | §3.7 |
| Multi-scala | spento per EoDT e baseline (`multi_scale: off`): ingresso quadrato fisso | deciso | §7.1 |
| Git | l'agente non fa commit: li faccio io a mano | deciso | §2 |
| Logging | W&B offline sul nodo di calcolo, sincronizzato dal nodo di login; stessa run tra job ripresi | deciso | §7.3 |

## 1. Obiettivo

EoDT (Encoder-only Detection Transformer) applica l'approccio di **EoMT** alla detection di **RF-DETR**: niente projector e niente decoder. Le query di detection vengono concatenate ai patch negli **ultimi L2 blocchi del ViT**, e un'unica attention fa sia la self-attention tra query sia la cross-attention verso l'immagine.

Il confronto con la baseline deve essere **alla pari**: stesso codice di training, stesso dataloader, stessa valutazione, stessa ricetta. Per questo EoDT vive dentro il fork di RF-DETR e riusa tutto ciò che non riguarda l'architettura.

## 2. Repository e regole

| Cartella | Ruolo | Permessi |
|---|---|---|
| `rf-detr-eodt/` | fork di RF-DETR, branch `eodt` | si modifica **solo** qui |
| `ref-rfdetr/` | RF-DETR originale, **stesso commit** del fork | sola lettura |
| `ref-eomt/` | EoMT originale | sola lettura |

- Tutto il codice nuovo va in `src/rfdetr/models/eodt/`, con i test in `tests/eodt/`.
- **I file esistenti di RF-DETR non si modificano**, salvo il punto minimo necessario a registrare il nuovo modello, a farlo passare nel loop di training e a costruire la baseline (vedi §8). Ogni modifica fuori da `eodt/` va proposta nel piano e approvata prima di farla.
- Non si aggiorna il fork da `upstream/develop` durante la tesi.
- **L'agente non fa commit, push o altre operazioni git che modificano la storia** (`commit`, `push`, `rebase`, `reset`, `merge`). Alla fine di ogni passo si ferma e riassume cosa ha cambiato e quali test passano. Rivedo il diff e faccio il commit a mano.

## 3. Architettura

```
immagine ─► patch embed (patch 16) + pos. embed + [CLS, 4 registri]
         ─► blocchi L1 (solo token del ViT)
                │
                ├─► query (§3.3): selection ─► Q query con box b0
                │                 oppure learned (come EoMT) ─► Q query, senza b0
                ▼
         ─► [query ‖ token del ViT] ─► blocco L2 #1 ─► testa ─► b0 + Δ1   (aux loss)
                                    ─► blocco L2 #2 ─► testa ─► b0 + Δ2   (aux loss)
                                    ─► ...
                                    ─► blocco L2 #N ─► testa ─► b0 + ΔN   (uscita finale)
```

Lo schema mostra la modalità `selection`. Con la modalità `learned` le query non dipendono da L1, e la testa predice il box direttamente, senza b0 (§3.3, §3.5).

### 3.1 Backbone

- DINOv2 **da timm**, con attention globale (nessuna finestra).
- Nome: `vit_small_patch14_reg4_dinov2` (ViT-S); `vit_base_patch14_reg4_dinov2` e `vit_large_patch14_reg4_dinov2` per le altre taglie.
- **Patch 16 al caricamento**: `timm.create_model(name, pretrained=True, img_size=(R, R), patch_size=16, num_classes=0)`. timm ricampiona il patch embedding (FlexiViT) e interpola il positional embedding sulla griglia R/16. È lo stesso schema di `ref-eomt/models/vit.py`.
- **Registri: sì** (deciso con Niccolò). DINOv2 con 4 registri, sia in EoDT sia nella baseline.
- **Risoluzione R = 576** (deciso), la stessa della baseline (configurazione Medium di RF-DETR, §5). Uguale per tutte le taglie. Si imposta nel file di configurazione (§3.7), mai nel codice. Ingresso quadrato fisso, quindi niente padding: con R = 576 la griglia è 36×36 = 1296 patch.
- Normalizzazione dell'immagine: la fa già il dataloader di RF-DETR (media e deviazione standard di ImageNet). **Non va ripetuta nel modello**: la normalizzazione interna di EoMT non va copiata.

### 3.2 Divisione L1 / L2

- L2 = numero di blocchi finali che contengono le query. **Valori di partenza: 3 per ViT-S e ViT-B, 4 per ViT-L**, come in EoMT. È oggetto di ablazione, quindi è **obbligatorio in ogni file YAML**: nessun default nel codice né tabella per taglia. Se manca, la configurazione dà errore. I valori di partenza sono scritti nei YAML di partenza.
- L1 = blocchi totali − L2. Nei blocchi L1 entrano solo i token del ViT (CLS, registri e patch).

### 3.3 Query: due modalità

La costruzione delle query è **opzionale e si sceglie da configurazione** con `query_mode`. Entrambe le modalità devono restare disponibili, perché il confronto tra le due è un esperimento.

| `query_mode` | Query | Box | `enc_outputs` |
|---|---|---|---|
| `learned` (come EoMT) | `nn.Embedding(Q, D)`, uguali per ogni immagine | predetto direttamente dalla testa | assente |
| `selection` (come RF-DETR) | contenuto learnable + encoding del riferimento | riferimento + Δ | presente |

Q è configurabile, default 300 (EoMT ne usa 200). Nella modalità `selection`, Q è il numero di patch selezionati (top-Q).

**Modalità `learned` (come EoMT).** Come in `ref-eomt/models/eomt.py` (`self.q`, concatenato in `EoMT.forward`): Q vettori imparati, che non dipendono dall'immagine, attaccati ai token del ViT prima del primo blocco L2. Non c'è b0: la testa predice il box assoluto, `sigmoid(MLP(query))`, come in DETR.

**Modalità `selection` (come RF-DETR).** Riprende la selection a due stadi di RF-DETR (`ref-rfdetr/src/rfdetr/models/transformer.py`, `Transformer.forward`, ramo `if self.two_stage`), adattata a **una sola scala**: i patch in uscita da L1.

1. Prendi i soli token dei patch in uscita da L1 e applica `vit.norm`, come RF-DETR applica la norm finale di DINOv2 a ogni feature intermedia esportata (`apply_layernorm`). La norm vale solo per questo ramo: in L2 entrano i token grezzi di L1.
2. Genera le proposte con `gen_encoder_output_proposals` (stesso file), usando una sola scala e `unsigmoid=False` (perché `bbox_reparam=True`).
3. Applica `Linear` + `LayerNorm` (l'equivalente di `enc_output` / `enc_output_norm`), poi la testa di classe (`enc_out_class_embed`).
4. **Top-Q** in base al massimo punteggio di classe.
5. Solo sui token selezionati, la testa del box (`enc_out_bbox_embed`) predice un Δ rispetto alla proposta, con la stessa formula `bbox_reparam` di RF-DETR:
   - centro = Δxy · wh_proposta + xy_proposta;
   - dimensioni = exp(Δwh) · wh_proposta.

   Il risultato è il **box iniziale b0**, che viene staccato dal grafo (`detach`) come in RF-DETR. Il detach riguarda solo b0: la loss del primo stadio arriva anche ai blocchi L1, come in RF-DETR arriva al backbone.
6. Come in RF-DETR, a b0 si applica un **offset imparato** `refpoint_embed` (`nn.Embedding(Q, 4)`, a zero all'inizio, indicizzato per posizione nel top-Q) con la stessa formula `bbox_reparam`: il risultato è il **riferimento**. All'inizio il riferimento coincide con b0.
7. Costruisci il token di ogni query come **contenuto learnable** (`nn.Embedding(Q, D)`, come `query_feat` di RF-DETR) + **encoding del riferimento**: `gen_sineembed_for_position(riferimento)` seguito da un MLP, come `ref_point_head` nel decoder di RF-DETR. I due termini si sommano una volta sola, all'ingresso di L2. In ogni blocco L2 la testa predice Δ rispetto al riferimento (come `lite_refpoint_refine` di RF-DETR); i box non sono limitati a [0, 1], come in RF-DETR.
8. Le uscite di questo stadio (classi e box b0 prima del detach e senza offset) vanno in `enc_outputs`, per la loss del primo stadio: stesso `SetCriterion`, matching ungherese sulle Q proposte, pesi `_enc` uguali a quelli finali, come in RF-DETR.

**Limite di risoluzione per `selection`:** `gen_encoder_output_proposals` azzera le proposte con coordinate fuori da (0,01; 0,99). Con griglia ≥ 50 (≥ 800 px a patch 16) l'anello esterno di patch ha proposte nulle: se uno di quei patch viene selezionato, il suo box ha dimensione 0 e non riceve gradiente. RF-DETR ha lo stesso comportamento, quindi non si aggiunge un validatore (si manterrebbe la parità con la baseline); alle risoluzioni previste (576, 640, 704) non succede. Da riconsiderare solo se si va a ≥ 800 px.

**Inizializzazione dell'ultimo strato di `bbox_embed`:** dipende dalla modalità. In `learned` il box è assoluto: inizializzazione di default di `nn.Linear`, come DETR (con lo zero tutte le query partirebbero dallo stesso box). In `selection` il box è Δ rispetto al riferimento: **peso e bias a zero**, come RF-DETR, così all'inizio ogni query predice il proprio riferimento (cioè b0, perché `refpoint_embed` parte da zero). Il bias di `class_embed` parte da una probabilità a priori di 0,01 in entrambe le modalità, costante nel codice come in `LWDETR`.

> **Ruolo delle due modalità:** `selection` è il **modello principale** di EoDT: la baseline RF-DETR usa la query selection, quindi solo con `selection` il confronto "baseline senza projector → EoDT" cambia il solo decoder. `learned` è un'**ablazione** nello stile di EoMT (query imparate, come Mask2Former ed EoMT): mostra se un detector encoder-only ha bisogno della query selection. Con `learned` la convergenza è più lenta (overfit del passo 5).
>
> **Ordine di implementazione:** prima `learned` (più semplice, usata per verificare il modello e l'infrastruttura: passi 1-6), poi `selection` (passo 9), prima del confronto con la baseline.

### 3.4 Blocchi L2

- Concatena **[query ‖ token del ViT]**, con le query in testa, come `ref-eomt/models/eomt.py`, `EoMT.forward`.
- Ogni blocco L2 è il **blocco originale di DINOv2**, con gli stessi pesi e la stessa struttura: attention, MLP e layer scale. **Nessun layer nuovo dentro i blocchi.**
- Attention: piena tra tutti i token nella versione base. L'attention mask serve solo per le varianti di Group-DETR (§4).
- Niente masked attention e niente mask annealing: sono specifici della segmentazione.

### 3.5 Testa e box

- **Una sola testa condivisa** da tutti i blocchi L2: `vit.norm` → classi (`Linear`) e box (MLP a 3 strati, come `bbox_embed` di RF-DETR).
- **La LayerNorm è la norm finale pre-addestrata del ViT** (`vit.norm`), applicata agli stati delle query dopo ogni blocco L2, come fa EoMT (`self.encoder.backbone.norm(x)` prima di `_predict`). Nessuna LayerNorm nuova: segue il principio della §3.4 (nessun layer nuovo dove il ViT ne ha già uno). Una LayerNorm dedicata alla testa resta un'eventuale ablazione.
- Si applica **dopo ogni blocco L2**, sui soli token delle query.
- Box, a seconda di `query_mode`:
  - `learned`: box assoluto, `sigmoid(bbox_embed(query))`, in ogni blocco L2;
  - `selection`: **Δ sempre rispetto al riferimento** (b0 + offset `refpoint_embed`, §3.3), non al box del blocco precedente. È il comportamento di RF-DETR con `lite_refpoint_refine=True` e `bbox_reparam=True` (vedi `LWDETR.forward` in `ref-rfdetr/src/rfdetr/models/lwdetr.py`).
- Uscita in formato RF-DETR, così `SetCriterion` e `PostProcess` funzionano senza modifiche:

```python
{
    "pred_logits": Tensor[B, Q, C],   # ultimo blocco L2
    "pred_boxes":  Tensor[B, Q, 4],   # cxcywh normalizzati in [0, 1]
    "aux_outputs": [{"pred_logits": ..., "pred_boxes": ...}, ...],  # gli altri blocchi L2
    "enc_outputs": {"pred_logits": ..., "pred_boxes": ...},         # solo con query_mode = "selection"
}
```

### 3.6 Scaling da ViT-S a ViT-B e ViT-L

**[DA DECIDERE]** Come scalare la baseline (decoder uguale per tutte le taglie oppure scalato come in RF-DETR) non è ancora deciso. Per EoDT lo scaling è chiaro: cambia il backbone e, al massimo, L2.

**Il codice deve permettere di passare da una taglia all'altra cambiando solo la configurazione, senza toccare il codice.** Devono quindi essere parametri di configurazione, senza valori scritti nel codice:
- **EoDT:** nome del backbone timm, L2, risoluzione, numero di query; la dimensione nascosta si ricava dal backbone;
- **baseline:** taglia del backbone DINOv2, layer del decoder, dimensione nascosta (`hidden_dim`), teste di attention, numero di finestre, projector sì/no, risoluzione.

Il test del passo 1 e quello del passo 4 (§9) vanno ripetuti anche con ViT-B e ViT-L, almeno in forward e backward, senza training.

### 3.7 Configurazione ed esperimenti

**Regola generale:** ogni valore che può cambiare tra un esperimento e l'altro è un parametro di configurazione, mai un valore scritto nel codice. Ogni esperimento corrisponde a un file di configurazione, così da poter tenere traccia di cosa è stato lanciato.

- **Formato:** lo stesso dei YAML di RF-DETR in `configs/` (`model_config` e `train_config` con `class_path` e `init_args`), con `EoDTConfig` come `ModelConfig` (§8 punto 2).
- **Lettura:** un loader in `src/rfdetr/models/eodt/` (pyyaml), **non** il comando `rfdetr fit`. Motivo: `rfdetr fit` (`LightningCLI`) richiede `jsonargparse`, non installato, e soprattutto costruisce il `Trainer` senza passare da `build_trainer`, quindi epoch, gradient clipping, precisione, EMA, valutazione COCO e checkpoint di `TrainConfig` non verrebbero applicati. Il training usa `build_trainer` + `RFDETRModelModule` + `RFDETRDataModule`, come `RFDETR.train()`. Lo stesso loader vale per la baseline.
- **Dove:** `configs/eodt/` per EoDT e `configs/baseline/` per la baseline, un file per esperimento, con un nome che dica cosa cambia (per esempio `eodt_vits_576_l2-3_learned.yaml`, `baseline_medium_nowin.yaml`).
- **Parametri da esporre almeno:**
  - **modello EoDT:** backbone (che determina anche i registri: `…_reg4_dinov2` = 4 registri, nessun parametro separato), risoluzione, L2, numero di query, `query_mode` (`learned` / `selection`), numero di gruppi (`group_detr`: 1 = alternativa (a), > 1 = alternativa (b));
  - **baseline:** tutti quelli della §3.6;
  - **training:** epoch, batch, learning rate (testa e backbone), decadimento per strati, weight decay, warm-up (§7), EMA, `cls_loss_coef`, seed.
  - **non esposti:** i coefficienti di box (5) e GIoU (2) e i costi del matcher (2, 5, 2) restano fissi ai valori di RF-DETR (`ModelDefaults`), identici per EoDT e baseline (§7.1). Si espongono solo se la rifinitura della §7.2 lo richiede, con un test di equivalenza con `ref-rfdetr`.
- **Tracciabilità:** ogni run salva nella propria cartella di output una copia della configurazione completa usata, compresi i default. Senza `LightningCLI` non c'è un salvataggio automatico: lo fa il codice di training (passo 6). Nella stessa cartella vanno salvati anche il commit git del codice e le versioni di Python, PyTorch, CUDA e timm.
- **Default:** i valori di §3, §5 e §7 sono i default delle configurazioni. Un esperimento li cambia solo nel proprio file YAML.
- **Test:** per ogni parametro esposto, verificare che un valore diverso nel YAML arrivi davvero al modello (per esempio: con risoluzione 640 la griglia diventa 40×40; con L2 = 4 le query entrano un blocco prima).

## 4. Group-DETR

Le alternative da confrontare sono due, scelte con `group_detr` nel file di configurazione:
- **(a) niente gruppi** (`group_detr = 1`): Q query, nessuna duplicazione. È anche **la prima versione** da implementare, con entrambe le modalità di query, e non richiede codice in più.
- **(b) patch isolati** (`group_detr = G > 1`), da implementare **dopo** che la modalità `selection` funziona: G gruppi da Q query. Con l'attention mask:
    - le query di un gruppo vedono solo le proprie query e i patch;
    - i patch non vedono le query.

  Così i gruppi non si contaminano passando dai patch. Ogni gruppo ha le proprie query (con `selection`, la propria query selection, come in RF-DETR). Deve funzionare con entrambe le modalità. In valutazione si usa solo il primo gruppo.
- Scartata: duplicare i patch per ogni gruppo, perché troppo costosa.

## 5. Baseline

La baseline è **RF-DETR riaddestrato con lo stesso training di EoDT**, senza Group-DETR (deciso con Niccolò, per contenere il costo). Si toglie un componente alla volta fino ad arrivare a EoDT, come nella Tabella 1 di EoMT:

| Run | Finestre | Projector | Decoder |
|---|---|---|---|
| RF-DETR | sì (2) | sì | sì |
| senza finestre | no | sì | sì |
| senza projector | no | no: il decoder legge l'uscita dell'ultimo blocco del ViT, portata a `hidden_dim` con `Linear` + `LayerNorm` | sì |
| **EoDT** | no | no | no: query negli ultimi L2 blocchi |

- **Tutte le run vanno riaddestrate**, compresa quella con le finestre: i training devono essere equivalenti (Niccolò).
- **Configurazione:** **Medium di RF-DETR** sul ViT-S (deciso al passo 7): 576 px, 4 layer del decoder, 300 query. Niccolò ha indicato di partire da una configurazione di default di RF-DETR. Tutti i parametri del decoder (layer, dimensione nascosta, teste, punti) sono scritti esplicitamente nei YAML della baseline, così un cambio futuro (per esempio lo scaling a ViT-B e ViT-L, §3.6) richiede solo il YAML. `dim_feedforward` resta 2048, fisso come in tutte le taglie di RF-DETR.
- **Senza projector:** il decoder legge **solo l'uscita dell'ultimo blocco** del ViT (`out_feature_indexes: [12]` per ViT-S), portata a `hidden_dim` con **`Linear` + `LayerNorm`**: si tolgono insieme la fusione dei 4 blocchi e il projector, cioè proprio ciò che EoDT elimina. La `LayerNorm` resta perché il `MultiScaleProjector` di RF-DETR termina con una LayerNorm: senza, l'anello toglierebbe anche la normalizzazione dell'ingresso del decoder, e non si saprebbe quale delle due cose ha pesato. Così misura solo il contributo della fusione multi-scala. È la proiezione minima standard nei modelli tipo DETR.
- In comune con EoDT: DINOv2 con 4 registri, patch 16 (§8 punto 4), risoluzione, `group_detr = 1`, niente NAS, niente Objects365, stessa ricetta e stesse epoch (§7).
- **Latenza dell'RF-DETR originale:** si misura anche sui checkpoint pubblicati, senza riaddestrare (§6). Serve perché senza finestre la baseline è più lenta dell'originale, e confrontarsi solo con lei favorirebbe EoDT.

## 6. Valutazione

- **Accuratezza:** AP COCO sul validation set 2017 (5.000 immagini), con la valutazione di RF-DETR con `eval_max_dets: 100` (COCO standard; default di RF-DETR 500) e `num_select: 300` (come RF-DETR e la famiglia DINO). Nota: in COCOeval `maxDets` vale **per coppia (immagine, classe)**, quindi con `num_select: 300` si valutano fino a 300 detection per immagine. È la stessa convenzione con cui RF-DETR e DINO riportano i loro numeri, quindi si mantiene per la confrontabilità con i paper. Uguale per EoDT e baseline. Prima del confronto con la Tabella 5 di RF-DETR (§7.2) va verificato con quale valore sono calcolati i numeri del paper. Il test set non si usa (non ha etichette pubbliche). Per COCO, anche la fase `test` di RF-DETR ricade su val2017. Quale checkpoint si valuta è indicato in §7.1.
- **AP per dimensione (small/medium/large):** registrate da RF-DETR dal passo 6a. **Attenzione:** RF-DETR non passa alla metrica le `area` delle annotazioni COCO (`_convert_targets` in `training/callbacks/coco_eval.py`), quindi la dimensione di un oggetto si calcola dall'area del box, non da quella della segmentazione come nel COCO ufficiale. Le AP per dimensione sono confrontabili tra EoDT e baseline, ma **non con quelle dei paper**. [DA DECIDERE] se correggerlo prima di confrontarsi con i paper. La mAP 50:95 complessiva non ne è toccata.
- **Velocità:** latenza (e FPS), **rimisurata da zero per tutti i modelli nelle stesse condizioni**, come fa EoMT (Appendice A.3): non si copiano i numeri dei paper, compresi quelli di RF-DETR (T4 + TensorRT), che quindi non sono confrontabili con i nostri. Protocollo in `configs/benchmark/a100.yaml`, script `scripts/eodt/benchmark.py`, job `scripts/leonardo/benchmark.sh`:
  - **A100 di Leonardo**, nodo riservato (`--exclusive`);
  - si misura il **forward di inferenza** (`export()`): dall'immagine già preprocessata e già sulla GPU a box e logit dell'ultimo blocco L2 o layer del decoder, senza teste intermedie né `enc_outputs` (il primo stadio di `selection` resta). Preprocessing, copia sulla GPU e post-processing esclusi (il post-processing è identico per tutti: 300 query, 91 classi);
  - `torch.compile` (modo default, `fullgraph=True`, **niente CUDA graphs**) e autocast bf16; SDPA in entrambi i ViT. Se un modello non si compila o non supporta bf16, si misura senza e lo si registra;
  - batch 1. La prima chiamata (compilazione) è cronometrata a parte, poi **500** iterazioni di warm-up, poi tutte le **5.000** immagini di val2017. Ogni forward tra due `torch.cuda.synchronize()`, cronometrato con `perf_counter`. Si riporta la media, con mediana, deviazione standard e p95; FPS = 1000 / media. Differenze di latenza sotto il 2% non si presentano come risultati;
  - modelli: EoDT (`selection`, `learned`), i tre anelli della baseline e i checkpoint pubblicati **Nano, Small, Medium, Large**, costruiti con un solo gruppo di query (parametri usati in inferenza). I pubblicati non hanno registri (4 token su 1.296, effetto trascurabile, da dichiarare); le loro AP vengono dal paper con Objects365 e non sono confrontabili con le nostre run: nel grafico latenza-accuratezza vanno distinte;
  - memoria: `max_memory_allocated` durante la misura, pesi compresi;
  - **MAC** (FLOPs / 2, convenzione fvcore/detectron2) con `FlopCounterMode` di PyTorch, su un'immagine in fp32 con il backend math di SDPA (su CPU l'attention fused non viene contata), totali e per componente. Non conta `grid_sample` (~0,003 GMAC sulla baseline), softmax (~0,5%, uguale per tutti), normalizzazioni ed elementwise;
  - con i risultati si salvano protocollo, nodo, GPU, versioni e commit.
- **Script di benchmark:** uno solo, usato per EoDT, per la baseline e per i checkpoint pubblicati di RF-DETR (§9 passo 13). La latenza non dipende dai pesi addestrati: i numeri definitivi si misurano sui modelli finali.

## 7. Training

### 7.0 Impostazione

- **Una sola ricetta, identica per EoDT e baseline.** Si parte dalla ricetta di RF-DETR e la si rifinisce; la ricetta finale si usa per tutti i modelli.
- **Attenzione:** i default di `TrainConfig` sono pensati per il **fine-tuning** di un RF-DETR già addestrato (`pretrain_weights` = checkpoint pubblicato): learning rate costante, nessun warm-up, checkpoint migliore su val. Qui invece si parte da DINOv2 con parti nuove casuali (decoder nella baseline, query e teste in EoDT), come in EoMT. Per questo il punto di partenza ha **due sole correzioni**, entrambe già disponibili in RF-DETR:
  - un **warm-up** (`warmup_epochs`);
  - un **learning rate che scende verso la fine** (`lr_scheduler: "cosine"`), così il training finisce in un punto stabile.
- **Dataset:** COCO 2017, train (118.287 immagini); valutazione su val2017 (§6).
- **Multi-scala spento** per EoDT e baseline: `multi_scale: off` e `square_resize_div_64: true` in ogni YAML, e il builder di EoDT rifiuta valori diversi. Motivo: EoDT ha un ingresso quadrato fisso (pos embed per una sola griglia), come EoMT. È una differenza rispetto alla ricetta originale di RF-DETR (multi-scala per batch, per la Medium da 416 a 736 px), uguale per entrambi i modelli e da dichiarare nella tesi. Il multi-scala per entrambi resta una possibile rifinitura della §7.2, con `dynamic_img_size` di timm in EoDT.
- **Augmentation:** pipeline di torchvision, fissata esplicitamente con `augmentation_backend: torchvision` in ogni YAML. Albumentations non è installato (§11); fissarlo evita che il comportamento cambi a seconda dei pacchetti presenti.
- **Niente NAS e niente Group-DETR** nella prima versione.
- **Niente pre-training su Objects365** (deciso): EoDT e baseline partono entrambi da DINOv2 e si addestrano solo su COCO.

### 7.1 Ricetta di partenza

| Iperparametro | RF-DETR (default) | EoMT | Partenza | Stato |
|---|---|---|---|---|
| Learning rate | 1e-4 | 1e-4 | 1e-4 | deciso |
| Learning rate del backbone | 1,5e-4 | 1e-4 | 1,5e-4 | deciso |
| Decadimento per strati | 0,8 | 0,8 | 0,8, con la formula sotto la tabella | deciso |
| Weight decay | 1e-4 | 0,05 | 1e-4 | deciso |
| Gradient clipping | 0,1 | 0,01 | 0,1 | deciso |
| Loss, matcher, augmentation | RF-DETR | — | RF-DETR | deciso |
| EMA | sì (0,993) | no | sì | deciso |
| Precisione | bf16 su A100 | fp16 | bf16 | deciso |
| Warm-up | nessuno | due fasi | lineare, `warmup_epochs: 1` (circa 7.400 iterazioni con batch 16) | partenza, da rifinire |
| Schedule del learning rate | costante (`step` con `lr_drop` = epoch) | polinomiale fino a 0 | `cosine` fino a ~0 | da rifinire |
| Epoch | 100 | 12 | **12**, fisse, uguali per tutti | partenza; il numero definitivo dopo le prime run |
| Batch totale | 4 per GPU (16 effettivo raccomandato) | 16 | **16**: `batch_size: 4` per GPU × 4 GPU, `grad_accum_steps: 1` | partenza |
| Seed | — | 0 | **0**, uguale per EoDT e baseline | deciso |
| Checkpoint riportato | migliore su val (`checkpoint_best_total.pth`) | finale | proposta: finale con EMA (`last_ema.pth`), migliore registrato solo come controllo | [DA DECIDERE] |

**Learning rate e weight decay del ViT: identici alla baseline**, tensore per tensore (formula di RF-DETR: `get_dinov2_lr_decay_rate`, `get_dinov2_weight_decay_rate`, `lr_component_decay`²):
- learning rate = `lr_encoder` · 0,8^(n + 1 − id) · 0,7², con n = profondità + 1 (13 per ViT-S e ViT-B, 25 per ViT-L) e id = 0 per gli embedding (patch, pos, CLS, registri), i + 1 per il blocco i, n + 1 per la norm finale. Quindi ultimo blocco ×0,8², embedding ×0,8¹⁴;
- **i blocchi L2 seguono la stessa formula degli altri blocchi**: sono blocchi pre-addestrati di DINOv2, non strati nuovi;
- weight decay 0 su embedding, bias e norm; `weight_decay` sugli altri pesi, layer scale compreso;
- moduli nuovi: learning rate `lr` = 1e-4, come query e teste di RF-DETR. Sono: query (`learned`) o `query_feat`, `refpoint_embed` e `ref_point_head` (`selection`), `class_embed`, `bbox_embed` e il primo stadio della selezione (`enc_output`, `enc_output_norm`, `enc_out_class_embed`, `enc_out_bbox_embed`). Unica differenza consapevole da RF-DETR: lì `ref_point_head` sta nel decoder e prende `lr` · 0,7 (`lr_component_decay`); in EoDT il decoder non c'è, e `ref_point_head` è un modulo nuovo come le query, quindi prende `lr`.

**Valori di partenza** (confronto con la baseline, passo 10): batch totale 16, 12 epoch, `warmup_epochs: 1`, seed 0. Non sono definitivi: si rifiniscono al passo 11 (§7.2). Note:
- la schedule `cosine` dipende dal numero di epoch: run con epoch diverse non sono confrontabili tra loro, solo EoDT e baseline alla stessa durata;
- il YAML con questi valori è quello per Leonardo (bf16, 4 GPU). Per l'overfit in locale (passo 5) c'è un YAML a parte (`eodt_vits_576_l2-3_learned_overfit.yaml`): stesso modello a **576 px** (memoria misurata sulla GTX 1650: 3,36 GB su 4), fp32, batch 4 (batch effettivo 4, perché con 20 immagini un batch più grande fa campionare con ripetizione), 120 epoch con 10 di warm-up (stessa forma della schedule), augmentation spenta (`aug_config: {}`, `scale_jitter: false`), valutazione di modello ed EMA. Se va in out of memory si riduce prima il batch, non la risoluzione.

Tutti i valori si impostano nei file YAML (§3.7). Il checkpoint riportato, qualunque sia, deve seguire la stessa regola per EoDT e baseline.

- **Warm-up in due fasi di EoMT** (prima solo le parti nuove, poi anche il backbone; `ref-eomt/training/two_stage_warmup_poly_schedule.py`): opzione da configurazione, **spenta di default**. Si prova solo se il warm-up lineare non basta. Se si usa, si usa anche nella baseline.

### 7.2 Rifinitura della ricetta

- **Si rifinisce solo sul ViT-S.** La ricetta trovata si porta su ViT-B e ViT-L senza nuove ricerche.
- **Pochi parametri, uno alla volta**, in quest'ordine:
  1. **stabilità** (warm-up, learning rate, gradient clipping), se la loss esplode o l'AP non sale;
  2. **durata** (epoch, schedule), guardando le curve di val.
- **Non sono rifiniture** le scelte di EoDT (L2, `query_mode`, Group-DETR): sono gli esperimenti della tesi e si fanno con la ricetta già fissata.
- **Parità con la baseline:** ogni modifica alla ricetta va provata anche sulla baseline. Se una modifica aiuta EoDT ma peggiora la baseline, si tiene una sola ricetta e lo si riporta.
- **Tracciabilità:** ogni tentativo è un file YAML e una run su W&B. Si tiene l'elenco delle configurazioni provate e del motivo della scelta finale.
- **Controllo che la baseline non sia indebolita:** confrontare la baseline con finestre con il risultato di RF-DETR addestrato senza Objects365 (paper, Tabella 5), dopo aver verificato nel paper con quante epoch è stato ottenuto.

### 7.3 Logging su W&B da Leonardo

I nodi di calcolo di Leonardo non hanno accesso a internet: W&B lavora **offline** durante il job e i dati si caricano **dal nodo di login**.

- RF-DETR ha già il logger W&B (`TrainConfig`: `wandb`, `project`, `run`; `WandbLogger` in `training/trainer.py`). Va attivato da YAML, non scritto da capo.
- **Nello script SLURM:**
  - `WANDB_MODE=offline`;
  - `WANDB_DIR` su `$WORK` o `$SCRATCH`, non su `$HOME`, che ha poco spazio;
  - `WANDB_RUN_ID` fisso per esperimento e `WANDB_RESUME=allow`: un training che supera le 24 ore viene spezzato in più job, che devono continuare **la stessa run** su W&B invece di crearne una nuova per ogni job.
- **Dal nodo di login:** `wandb sync <cartella della run offline>` alla fine di ogni job, oppure periodicamente durante il job per seguire le curve. Il login a W&B lo faccio io a mano, una volta sola.
- **Cosa registrare:** loss (totale e per componente), AP e AP50/AP75/APs/APm/APl a ogni valutazione, learning rate, tempo per iterazione e immagini al secondo, memoria GPU. Configurazione completa e commit git come in §3.7.
- **Non caricare su W&B i checkpoint**: sono pesanti e restano su `$WORK`.
- Con più GPU scrive su W&B solo il processo principale (rank 0). Lightning lo fa già: verificare che non ci siano run doppie.
- Test (passo 6): un job di prova di pochi minuti, poi sincronizzato dal nodo di login, deve comparire su W&B con configurazione e curve; un secondo job ripreso dal checkpoint deve continuare la stessa run.

## 8. Punti di integrazione da verificare (il piano li deve risolvere)

1. **Learning rate per strati:** `_build_param_dicts` (`training/param_groups.py`) presuppone `model.backbone` di tipo `Joiner` con `backbone[0].get_named_param_lr_pairs(...)`, e cerca i parametri del decoder con il nome `transformer.decoder`. Per EoDT serve il decadimento per strati sui blocchi del ViT, compresi i blocchi L2. **Proporre** la soluzione meno invasiva: una struttura compatibile oppure una funzione dedicata per i gruppi di parametri.
2. **Costruzione del modello:** come `build_model` / `build_model_from_config` (`models/lwdetr.py`) e la configurazione (`config.py`) arrivano a `RFDETRModelModule`. EoDT ha bisogno di una propria `ModelConfig` (`EoDTConfig`) e di un punto di registrazione.
3. **Ingresso:** `LWDETR.forward` riceve un `NestedTensor`. EoDT deve accettare lo stesso tipo e ignorare la maschera, perché l'ingresso è quadrato e senza padding (`square_resize_div_64`).
4. **Backbone della baseline:** RF-DETR carica DINOv2 da Hugging Face (`models/backbone/dinov2.py`). I registri sono già supportati (`use_registers`, encoder `dinov2_registers_windowed_small`). Però:
   - con `patch_size != 14` **i pesi pre-addestrati non vengono caricati**, perché la repo pubblica si aspetta un checkpoint RF-DETR già a patch 16. Serve una conversione 14→16 al caricamento (ricampionamento del patch embedding, come fa timm), altrimenti la baseline parte da pesi casuali;
   - `EncoderName` (`config.py`) prevede i registri solo per la taglia small. **Risolto al passo 8a senza toccare `config.py`:** `BaselineConfig` accetta `dinov2_registers_windowed_{small,base,large}`.
   - **la baseline con registri oggi non si costruisce** (verificato al passo 4b): con transformers 5.17, `encoder: dinov2_registers_windowed_small` dà `ValueError: out_features and out_indices should have the same length`, perché `dinov2_with_registers_small.json` contiene `out_indices` e `DinoV2.__init__` sovrascrive solo `out_features`. **Risolto al passo 8a** con una riga in `dinov2.py` (`dino_config.pop("out_indices", None)`), con test; i pesi a patch 16 si caricano dopo `build_model` (`eodt/dinov2_weights.py`) con le stesse funzioni di timm, e coincidono tensore per tensore e nelle feature con quelli di EoDT (S, B e L).
5. **Baseline senza projector:** RF-DETR non ha un'opzione per togliere il projector. Serve un'opzione da configurazione che lo sostituisca con `Linear` + `LayerNorm` sull'uscita dell'ultimo blocco del ViT (§5).

6. **Pesi delle aux loss:** `build_criterion_and_postprocessors` (`models/lwdetr.py`) costruisce i pesi delle loss intermedie con `range(args.dec_layers - 1)`. EoDT non ha decoder: servono **L2 − 1** pesi, uno per ogni blocco L2 tranne l'ultimo. Senza questa correzione le `aux_outputs` restano senza peso e la deep supervision sparisce **senza errori**. Va coperto da un test.
7. **Numero di classi:** `EoDT(num_classes=...)` è il numero di uscite. Come in `build_model`, il costruttore che legge la configurazione passa `num_classes + 1`: per COCO `ModelConfig.num_classes = 90` → **91 uscite** (gli id COCO 1-90 sono usati direttamente come indici; sigmoid per classe, nessuna classe "nessun oggetto").
8. **`query_mode` in `EoDTConfig`:** campo della configurazione già dal passo 4, per ora con il solo valore `learned`, così compare nei YAML delle prime run (passo 6). Il valore `selection` si aggiunge al passo 9.
9. **`C2f.forward` del projector con `torch.compile` (passo 13):** in `models/backbone/projector.py` la riga `y.extend(m(y[-1]) for m in self.m)` viene compilata da PyTorch 2.6 valutando il generatore prima dell'`extend`: i bottleneck ricevono tutti lo stesso ingresso invece di essere in catena, e il modello compilato dà uscite diverse dall'eager (fino a 0,97 sui box). Il training non è toccato (compile spento), ma il benchmark misurerebbe un'altra funzione. Correzione: ciclo `for` con `append`, uscita in eager identica bit per bit (il test di equivalenza con `ref-rfdetr` resta valido), compilato = eager; coperta da un test. Riguarda anche la baseline e i checkpoint pubblicati, quindi la latenza di RF-DETR si misura sul codice corretto. Possibile segnalazione a RF-DETR upstream.

Per i punti 4 e 5, proporre la soluzione meno invasiva.

## 9. Ordine di lavoro e verifiche

Ogni passo si chiude **solo** quando il suo test passa. Poi l'agente si ferma: rivedo il diff e faccio io il commit (§2) prima di passare al passo successivo.

**Principio dell'ordine:** su EoDT da solo si fanno i test, l'overfit e i job di prova di **verifica** su Leonardo (la run completa di verifica con `learned` è stata tolta il 05/11/2026, passo 6). La **rifinitura della ricetta** (§7.2) si fa solo dopo, guardando insieme EoDT e baseline, così la ricetta non viene tarata su un solo modello.

| # | Passo | Test di verifica |
|---|---|---|
| | **EoDT `learned`** | |
| 1 | Backbone timm a patch 16 | Nessuna chiave mancante o inattesa nel caricamento. Feature a patch 14 identiche al modello timm originale. Shape della griglia uguale a R/16. |
| 2 | Blocchi L2 con query | Shape corrette. **I token del ViT in uscita da L1 sono identici a quelli del ViT senza query.** Con 0 query, l'uscita è identica al ViT originale. |
| 3 | Query `learned` e testa | Formato dell'uscita come in §3.5, senza `enc_outputs`. Box in [0, 1]. |
| 4 | Collegamento a criterion, training e configurazione YAML | Forward e backward su 2 immagini. Loss finita. **Gradiente non nullo su tutti i parametri addestrabili.** Il modello si costruisce da un file in `configs/eodt/`. |
| 5 | **Overfit su 20 immagini** COCO, in locale | Dopo **2.400 step** (480 epoch, warm-up 40, valutazione ogni 4 epoch), sia per il modello sia per l'EMA: **AP50 ≥ 0,95** e **mAP 50:95 ≥ 0,85**; loss sempre finita; `loss_giou` dell'ultimo blocco ≤ 0,10. Soglie fissate prima del lancio. **Se non passa, ci si ferma e si fa debug: non si va avanti.** L'overfit si ripete poi su Leonardo (bf16, A100) come primo job del passo 6. **Esito (passo 5): PASS.** Con 600 step non bastava (mAP 0,795, convergenza lenta delle query `learned`; su 1 immagine mAP 1,0 dopo 215 step, nessun bug trovato); con 2.400 step: modello mAP 50:95 0,979 e AP50 0,988, EMA 0,977 e 0,988, `loss_giou` 0,013. |
| 6 | Script per Leonardo e **verifica dell'infrastruttura** su COCO (ViT-S, ricetta di partenza §7.1). **Aggiornato il 05/11/2026:** la run completa con `learned` si salta. `learned` è un'ablazione da rifare comunque con la ricetta finale (passo 11), quindi una run di 12 epoch servirebbe solo come verifica, e per la verifica bastano, **per ciascuno dei due modelli del passo 10** (EoDT `selection` e `baseline_medium_win`), un overfit in bf16 su A100 e i job di prova (stop e ripresa). Dopo i job di prova si passa direttamente al passo 10 | Il run parte, la loss scende e l'AP sale, la valutazione gira a fine epoch, configurazione e commit salvati nella cartella di output (§3.7). La ripresa da checkpoint funziona (limite di 24 ore per job). Logging su W&B come in §7.3. Prima della run lunga, un **job di prova di 10-15 minuti** a risoluzione piena su più GPU (lo lancio io). **Non si modifica la ricetta in base a questa run.** Da risolvere prima della run, emersi al passo 4c: (1) **seed con DDP**: `seed_everything(seed + rank)` dà a ogni rank un seed diverso del `DistributedSampler`, quindi i 4 rank non si dividono il dataset (circa il 68% di immagini distinte per epoch, stima del subagent): va corretto, anche per la baseline, con un test; (2) query e teste si inizializzano prima del seed; (3) SLURM con `--ntasks-per-node=4` e `srun`; (4) pesi DINOv2 nella cache su `$WORK` e `HF_HUB_OFFLINE=1`; (5) RAM con 4×(7+7) worker; (6) un `output_dir` già usato non deve essere sovrascritto; (7) bf16 verificato nel job di prova. |
| | **Baseline** | |
| 7 | **Decisione sulla baseline** con Niccolò (compito mio, non dell'agente): configurazione, finestre, quale run lanciare per prima | §5 aggiornata, [DA CONFERMARE] risolti, file YAML in `configs/baseline/`. |
| 8 | Implementazione della baseline (§5, §8 punti 4 e 5): conversione DINOv2 14→16 con registri, opzione senza projector. **Il codice si può scrivere anche prima del passo 7**, perché risoluzione e layer del decoder sono solo valori nel YAML. | Caricamento DINOv2 a patch 16 con registri senza chiavi mancanti. Tutte le run della catena si costruiscono solo da YAML. **Test di equivalenza** tra il fork e `ref-rfdetr/` (stesso checkpoint, stessa immagine, output identico) sulla configurazione originale, per verificare che le modifiche non abbiano cambiato RF-DETR. Overfit su 20 immagini anche per la baseline. **Esito (passo 8c): PASS.** Overfit della baseline senza finestre (`baseline_medium_nowin_overfit.yaml`) con setup e criteri del passo 5 (2.400 step, 71 min sulla GTX 1650, picco 2,95 GB): modello ed EMA con mAP 50:95 1,000 e AP50 1,000; `loss_giou` dell'ultimo layer del decoder 0,0015. **Esito (passo 8d): PASS.** Overfit della baseline con finestre (`baseline_medium_win_overfit.yaml`, la baseline del passo 10), stesso setup e criteri (2.400 step, 60,5 min, picco 2,96 GB): modello ed EMA con mAP 50:95 1,000 e AP50 1,000; `loss_giou` 0,0016. La baseline converge più in fretta di EoDT `learned` (mAP 50:95 ≥ 0,85 intorno all'epoch 185): atteso, perché le query passano per un decoder dedicato; non è un confronto di prestazioni. In locale, su Windows, serve `PYTHONIOENCODING=utf-8` (la tabella di `rich` non si scrive in cp1252). |
| | **EoDT `selection` (modello principale)** | |
| 9 | Query `selection` | Top-Q corretto. b0 senza gradiente. Con Δ = 0 il box è uguale al riferimento (uguale a b0 con `refpoint_embed` a zero, cioè all'inizio). `enc_outputs` presente. Il cambio di `query_mode` avviene solo da YAML. Ripetere i test 4 e 5 (compreso l'overfit su coco_mini con gli stessi criteri), e verificare che `learned` funzioni ancora. **Esito (passo 9): PASS.** 9a: 228 test veloci e 27 `gpu`, equivalenza del primo stadio con il `Transformer` di RF-DETR (box identici). 9b: overfit di `eodt_vits_576_l2-3_selection_overfit.yaml` con setup e criteri del passo 5 (2.400 step, 65 min sulla GTX 1650, picco 3,01 GB): modello mAP 50:95 0,959 e AP50 0,964, EMA 0,992 e 1,000; `loss_giou` dell'ultimo blocco 0,0033. Nelle ultime 40 epoch il modello senza EMA oscilla (picchi in `loss_ce`, mAP 0,99 → 0,96) anche con learning rate quasi nullo; l'EMA resta stabile. Causa non nota. L'ipotesi "top-Q discreto + `query_feat` indicizzato per posizione nel ranking" è indebolita dagli overfit 8c e 8d: anche la baseline ha lo stesso meccanismo e non oscilla. Da tenere d'occhio nelle run del passo 10 (picchi di `loss_ce` su W&B), confrontando con la baseline. |
| | **Confronto e ricetta** | |
| 10 | **Due run insieme**: EoDT `selection` e baseline **RF-DETR con finestre** (primo anello della catena, riaddestrato), ViT-S, stessa ricetta di partenza e stesse epoch. La ricetta si rifinisce (passo 11) sul riferimento vero, non su una baseline già privata di un componente. Gli anelli intermedi (senza finestre, senza projector) e l'ablazione `learned` si lanciano solo con la ricetta finale, dopo il passo 11 | Entrambe completate, curve su W&B confrontabili (stesse metriche, stessi intervalli di valutazione). |
| 11 | **Rifinitura della ricetta** (§7.2), guardando entrambe le run | Ogni modifica provata su EoDT e baseline. Ricetta finale scritta in §7.1, con l'elenco delle configurazioni provate. La run `learned` (ablazione) si rifà con la ricetta finale. |
| | **Resto di EoDT** | |
| 12 | Group-DETR, variante (b) | Attention mask: i patch non vedono le query, i gruppi non si vedono tra loro. Ripetere i test 4 e 5. |
| 13 | Script di benchmark (§6). **Fatto in parallelo al passo 10** (la latenza non dipende dai pesi) | Stessa misura su EoDT, baseline e un checkpoint pubblicato di RF-DETR. Due job di `benchmark.sh` sulla A100, poi `benchmark.py compare`: parametri e MAC identici, **latenze medie entro il 2%** (soglia fissata prima del lancio). Esito locale: script funzionante su EoDT `selection`, baseline con finestre e Medium pubblicato; compilato uguale al non compilato (dopo la correzione di §8 punto 9). MAC con il forward di inferenza: EoDT `selection` 47,76 (L1 32,80 + L2 14,40 + primo stadio 0,46 + teste 0,10), baseline senza finestre 47,94 (ViT 43,61 + projector 1,87 + primo stadio 0,16 + decoder 2,26 + teste 0,05), baseline con finestre 39,64. Ripetibilità sulla GTX 1650 solo indicativa: EoDT 0,3%, baseline 8,4%, Medium 4,2% (i modelli RF-DETR oscillano di più). Test sulla A100: da fare. |

## 10. Fuori dalla specifica (per ora)

Export, segmentazione, keypoint, modelli Plus, CUDA graphs, FP8, NAS, Objects365.

## 11. Ambienti

Le versioni sono fissate in `requirements-eodt.txt` (generato da `requirements-eodt.in`), **uguale per PC e Leonardo**: Python 3.11, PyTorch 2.6.0 con CUDA 12.4, torchvision 0.21.0, timm 1.0.30, transformers 5.17.0, pytorch-lightning 2.6.6, torchmetrics 1.8.2.

| | PC (test, passi 1-5) | Leonardo (training) |
|---|---|---|
| Cartella | `C:/dev/tesi/rf-detr-eodt` | `/leonardo_work/IscrC_EoDT/tcentonz/rf-detr-eodt` |
| Dati | `C:/dev/tesi/data/coco_mini` (20 immagini, overfit) | COCO 2017 completo in `/leonardo_scratch/fast/IscrC_EoDT/coco` (`$FAST`: veloce, nessuna cancellazione per inattività) |
| Output | — | `/leonardo_work/IscrC_EoDT/tcentonz/runs/<nome esperimento>` |
| Ambiente | `.venv` (uv) | `/leonardo_work/IscrC_EoDT/tcentonz/eodt-venv` |
| Attivazione | `.venv\Scripts\python.exe ...` | `module load python cuda && source /leonardo_work/IscrC_EoDT/tcentonz/eodt-venv/bin/activate` |
| GPU | GTX 1650, 4 GB, **niente bf16** (fp32 o fp16) | 4× A100 64 GB per nodo |
| Chi lancia | l'agente (test) | io, a mano (l'agente scrive gli script SLURM) |

- **Nuove dipendenze:** solo con il mio consenso. Si aggiungono a `requirements-eodt.in` e si rigenera il file bloccato con `python -m uv pip compile pyproject.toml requirements-eodt.in --extra train --universal --python-version 3.11 -o requirements-eodt.txt`; poi si reinstalla su PC e Leonardo.
- **PyTorch sul PC:** va preso dall'indice CUDA (`--index-url https://download.pytorch.org/whl/cu124`), perché quello di PyPI per Windows è solo CPU. Su Linux quello di PyPI è già CUDA 12.4.
- **Leonardo senza internet sui nodi di calcolo:** pesi di DINOv2 (timm/Hugging Face) e dataset vanno scaricati prima, dal nodo di login, nella cache su `$WORK`.
