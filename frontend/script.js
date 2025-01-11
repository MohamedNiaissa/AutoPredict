async function submitForm(event) {
    console.log("Submitting form...");
    event.preventDefault(); // Prevent default form submission behavior

    // Get form data
    const formData = new FormData(event.target);
    const brand = formData.get('brand');
    const year = parseInt(formData.get('year'));
    const km_driven = parseInt(formData.get('km_driven'));
    const fuel = formData.get('fuel');
    const seller_type = formData.get('seller_type');
    const transmission = formData.get('transmission');
    const owner = formData.get('owner');

    // Create the request body object
    const requestBody = {
        year: year,
        km_driven: km_driven,
        fuel: fuel,
        seller_type: seller_type,
        transmission: transmission,
        owner: owner,
        brand: brand
    };

    try {
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody),
        });

        if (response.ok) {
            const result = await response.json(); // Parse JSON from the backend
            console.log(result)
            const encodedResult = encodeURIComponent(JSON.stringify(result));
            window.location.href = `result.html?data=${encodedResult}`;
            document.getElementById('result').innerHTML = `
                <h4>Submission Success!</h4>
                <h5>API Response:</h5>
                <pre>${JSON.stringify(result.predicted_selling_price, null, 2)}</pre>
            `;
        } else {
            document.getElementById('result').innerHTML = `
                <h4>Error</h4>
                <p>There was an issue with the submission. Please try again.</p>
            `;
        }
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('result').innerHTML = `
            <h4>Error</h4>
            <p>An unexpected error occurred. Please try again later.</p>
        `;
    }
}
