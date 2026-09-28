// One viewer for offline reports, Gradio, and saved website examples.
function mountVasa(host) {
  let state={run:null,selected:'final',follow:true,mask:true,view:'step'}, timer=null;
  const rootNow=()=>host.querySelector('.vasa-report');
  function stop(){if(timer!==null)clearInterval(timer);timer=null;const b=rootNow()?.querySelector('[data-action=play]');if(b){b.textContent='Play';b.setAttribute('aria-pressed','false');}}
  function showImage(root,source){
    const stage=root.querySelector('.stage'),next=source?.cloneNode(true),token={};
    stage._vasaTransition=token;
    if(next){
      const images=[...next.querySelectorAll('img')];
      Promise.all(images.map(img=>img.decode().catch(()=>{}))).then(()=>{
        if(stage._vasaTransition!==token||!stage.isConnected)return;
        const previous=stage.lastElementChild;
        const same=previous?.querySelector('img')?.src===next.querySelector('img')?.src;
        stage.replaceChildren(...(previous?[previous]:[]),next);
        if(same||matchMedia('(prefers-reduced-motion: reduce)').matches){previous?.remove();return;}
        const fade=next.animate([{opacity:0},{opacity:1}],{duration:200,easing:'ease-out'});
        fade.finished.then(()=>previous?.remove()).catch(()=>{});
      });
    }
    const kind=source?.dataset.kind || 'original';
    const descriptions={working:['Working mask','Accumulated segmentation after edits.'],candidates:['SAM3 candidates','Proposed regions from SAM3 — not yet the working mask.'],inspection:['Inspection view','Evidence being checked by the model.'],original:['Original image','Input image without a mask.']};
    const [label,description]=descriptions[kind] || descriptions.original;
    const status=root.querySelector('.view-status');status.dataset.kind=kind;
    status.querySelector('.view-badge').textContent=state.selected==='final' && kind==='working' && root.dataset.live!=='true'?'Final result':label;
    status.querySelector('.view-description').textContent=description;
  }
  function draw(){
    const root=rootNow();if(!root){stop();return;}
    root.querySelectorAll('[data-image]').forEach(slot=>{const asset=root.querySelector(`.asset-bank img[data-asset="${slot.dataset.image}"]`);if(asset){const img=asset.cloneNode(true);img.alt=slot.dataset.caption;slot.replaceWith(img);}});
    const live=root.dataset.live==='true',cards=[...root.querySelectorAll('article[data-step]')];
    if(state.run!==root.dataset.run){stop();state={run:root.dataset.run,selected:'final',follow:true,mask:true,view:'step'};}
    if(live)stop();
    if(state.follow)state.selected=live&&cards.length?String(cards.length-1):'final';
    const card=cards.find(c=>c.dataset.step===state.selected);
    cards.forEach(c=>c.hidden=c!==card);
    root.querySelector('.final-detail').hidden=!!card;
    root.querySelectorAll('[data-select]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.select===state.selected)));
    const working=card?.querySelector('.working-state figure');
    const output=card?.querySelector('.gallery figure');
    const source=!state.mask?root.querySelector('.original figure'):(state.view==='working'?working:output)||working||root.querySelector('.final-image figure')||root.querySelector('.original figure');
    showImage(root,source);
    const mask=root.querySelector('[data-action=mask]');mask.textContent=state.mask?'Show original':'Return to selected view';mask.setAttribute('aria-pressed',String(!state.mask));
    const workButton=root.querySelector('[data-action=working]'),stepButton=root.querySelector('[data-action=step-view]');
    workButton.hidden=stepButton.hidden=false;
    workButton.disabled=!card;
    workButton.textContent='Working mask';
    workButton.title=working?.dataset.kind==='original'?'No working mask yet — shows the original image':'Accumulated mask at this step';
    workButton.setAttribute('aria-pressed',String(state.mask&&state.view==='working'));
    stepButton.setAttribute('aria-pressed',String(state.mask&&state.view==='step'));stepButton.disabled=!output;
    const follow=root.querySelector('[data-action=follow]');follow.hidden=!live;follow.textContent=state.follow?'Following live':'Back to live';follow.setAttribute('aria-pressed',String(state.follow));
    root.querySelector('[data-action=previous]').disabled=!cards.length||state.selected==='0';
    root.querySelector('[data-action=next]').disabled=state.selected==='final'||!cards.length;
    const label=card?`Step ${Number(state.selected)+1} of ${cards.length}`:live?'Preparing…':root.dataset.failed?'Last available view':'Final result';
    root.querySelector('.selection-label').textContent=label;
    root.querySelector('.playback').hidden=live||!!root.dataset.failed;
    const slider=root.querySelector('input[type=range]');slider.max=cards.length;slider.value=state.selected==='final'?cards.length:Number(state.selected);slider.setAttribute('aria-valuetext',label);slider.disabled=!cards.length;
    const play=root.querySelector('[data-action=play]');play.disabled=!cards.length;play.textContent=timer===null?'Play':'Pause';play.setAttribute('aria-pressed',String(timer!==null));
    if(timer!==null){const list=root.querySelector('.step-list'),active=list.querySelector('[aria-pressed="true"]');if(active)list.scrollTop+=active.getBoundingClientRect().top-list.getBoundingClientRect().top;}
    if(state.follow&&live){const list=root.querySelector('.step-list');list.scrollTop=list.scrollHeight;}
  }
  function select(value){stop();state.selected=value;state.follow=false;state.mask=true;state.view='step';draw();}
  function play(){
    if(timer!==null){stop();return;}
    const root=rootNow();if(!root||root.dataset.live==='true')return;
    const count=root.querySelectorAll('article[data-step]').length;if(!count)return;
    state.follow=false;state.mask=true;state.view='step';
    if(state.selected==='final')state.selected='0';
    // Close details before playback so long text doesn't distract from the action.
    root.querySelectorAll('details[open]').forEach(d=>d.open=false);
    timer=setInterval(()=>{if(!host.isConnected){stop();return;}const n=Number(state.selected)+1;state.selected=n>=count?'final':String(n);if(state.selected==='final')stop();draw();},3000);
    draw();
  }
  host.addEventListener('click',event=>{
    const target=event.composedPath().find(n=>n instanceof Element&&n.closest('.vasa-report'));if(!target)return;
    const root=target.closest('.vasa-report'),button=target.closest('button');
    if(target.closest('summary'))stop();
    if(button){
      const action=button.dataset.action,count=root.querySelectorAll('article[data-step]').length;
      if(button.dataset.select!==undefined){select(button.dataset.select);return;}
      if(action==='play'){play();return;}
      if(action==='previous'){select(String(state.selected==='final'?count-1:Math.max(0,Number(state.selected)-1)));return;}
      if(action==='next'){select(Number(state.selected)+1>=count?'final':String(Number(state.selected)+1));return;}
      stop();
      if(action==='mask')state.mask=!state.mask;
      if(action==='working'||action==='step-view'){state.view=action==='working'?'working':'step';state.mask=true;}
      if(action==='follow'){state.follow=true;state.mask=true;state.view='step';}
      draw();
    }
    const figure=target.closest('.gallery figure');if(figure){stop();event.preventDefault();state.mask=true;showImage(root,figure);}
  },true);
  host.addEventListener('input',event=>{const slider=event.composedPath().find(n=>n instanceof Element&&n.matches('.vasa-report input[type=range]'));if(slider)select(Number(slider.value)>=Number(slider.max)?'final':slider.value);},true);
  let current=null,revision=null;
  const observer=new MutationObserver(()=>{const root=rootNow(),next=root?.dataset.run+':'+root?.dataset.revision;if(root!==current||next!==revision){current=root;revision=next;draw();}});
  observer.observe(host,{childList:true,subtree:true,attributes:true,attributeFilter:['data-revision','data-run']});
  current=rootNow();revision=current?.dataset.run+':'+current?.dataset.revision;draw();
}
