import React, {useState} from "react"
// import Object
// params
const ParamChecker = () => {

    // paramList = 
    const params = { attack: true, decay: true, sustain: true, release: true };

    return (
        <div>
        {/* <h3>Searchable Parameters</h3> */}
        <ul>
            {Object.keys(params).map((key) => (
            <div>
            <li key={key.id}>{key} {params[key]}</li>
            {/* <p>{params[key]}</p> */}
            </div>
            ))}
        </ul>
        </div>
    );
}

export default ParamChecker