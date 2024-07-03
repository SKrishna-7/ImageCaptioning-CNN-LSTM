
    const upload=document.querySelector('#generate-cap');
    const input =document.querySelector('.file');

    const img=document.querySelector('#show-img')

    const caption=document.querySelector('#caption')

    function setLoader(){
        caption.innerText='Generating....'
    }
    function RemoveLoader(){
        caption.innerText=''
    }


    input.addEventListener('change',e=>{
        e.preventDefault();
        caption.innerText='Ready to Caption..'
        console.log(e.target.files[0]);
        img.src=URL.createObjectURL(e.target.files[0])
    })

    console.log(caption.innerText);

    

    upload.addEventListener('submit',e=>{
	if (!input.files[0]){
		alert('Please Select Image to generate caption...')
	}
	else{

        e.preventDefault();
        const data=new FormData();
        data.append('file',input.files[0])
        console.log(data);
        setLoader();
        fetch('http://localhost:8002/predict',{
            method:'post',
            body:data
        })
        .then((res)=>{
            return res.json()
        }).then((data)=>{
            RemoveLoader()
            console.log(data);
            caption.innerText=data.caption;
        })
        .catch((err)=>{
            console.log(err);
        })
}
    })